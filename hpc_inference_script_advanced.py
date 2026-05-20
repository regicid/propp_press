from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import os
import re
import torch
import torch.nn as nn
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from propp_fr import (
    generate_tokens_df, get_embedding_tensor_from_tokens_df,
    generate_entities_df, add_features_to_entities, perform_coreference,
    extract_attributes, generate_characters_dict,
    save_entities_df, save_book_file,
    load_tokenizer_and_embedding_model, load_models,
)

def _move_modules(obj, device, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))

    if isinstance(obj, nn.Module):
        obj.to(device).eval()
        return

    if isinstance(obj, dict):
        for v in obj.values():
            _move_modules(v, device, seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _move_modules(v, device, seen)
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            if not k.startswith('_'):
                _move_modules(v, device, seen)

# ---------------------------------------------------------
# CPU WORKER: Only handles SpaCy (CPU-bound, no GIL issues)
# ---------------------------------------------------------
global_spacy_model = None

def init_spacy_worker():
    """Load SpaCy model into the isolated CPU worker process."""
    global global_spacy_model
    import spacy
    global_spacy_model = spacy.load("fr_dep_news_trf")
    global_spacy_model.max_length = 500000

def cpu_process_spacy(file_name, text_content):
    """Generates the tokens DataFrame strictly on the CPU."""
    try:
        tokens_df = generate_tokens_df(
            text_content, 
            global_spacy_model,
            max_char_sentence_length=500000 
        )
        return (file_name, text_content, tokens_df, None)
    except Exception as e:
        return (file_name, None, None, str(e))

# ---------------------------------------------------------
# GPU THREAD: Handles PyTorch (Thread-safe, shares 1 model)
# ---------------------------------------------------------
def gpu_thread_task(file_name, text_content, tokens_df, root_directory, ctx):
    """Executes the GPU-bound parts of the pipeline using the shared model."""
    try:
        with torch.inference_mode():
            tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(
                text_content, tokens_df, ctx['tokenizer'], ctx['embedding_model'],
                mini_batch_size=128
            )
            
            entities_df = generate_entities_df(
                tokens_df, tokens_embedding_tensor, ctx['mentions_detection_model'],
                batch_size=256
            )
            
            entities_df = add_features_to_entities(entities_df, tokens_df)
            
            entities_df = perform_coreference(
                entities_df, tokens_embedding_tensor, ctx['coreference_resolution_model'],
                batch_size=50000, propagate_coref=True, rule_based_postprocess=False
            )
            
        tokens_df = extract_attributes(entities_df, tokens_df)
        characters_dict = generate_characters_dict(tokens_df, entities_df)
        save_entities_df(entities_df, file_name, root_directory)
        save_book_file(characters_dict, file_name, root_directory)
        return (file_name, 'ok', None)
    except Exception as e:
        return (file_name, 'error', str(e))


def setup(device='cuda'):
    assert torch.cuda.is_available(), "CUDA not available"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    _, mentions_detection_model, coreference_resolution_model = load_models()
    tokenizer, embedding_model = load_tokenizer_and_embedding_model(
        mentions_detection_model['base_model_name']
    )

    embedding_model = embedding_model.to(device).eval()
    _move_modules(mentions_detection_model, device)
    _move_modules(coreference_resolution_model, device)

    return {
        'tokenizer': tokenizer,
        'embedding_model': embedding_model,
        'mentions_detection_model': mentions_detection_model,
        'coreference_resolution_model': coreference_resolution_model,
        'device': device,
    }

# ---------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------
def process_dataframe_advanced(df, text_columns, root_directory, id_column, num_cpu_workers=16, num_gpu_threads=16, separator='\n\n'):
    root = Path(root_directory)
    root.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(getattr(row, id_column)), separator.join(getattr(row, c) for c in text_columns))
        for row in df.itertuples(index=False)
    ]

    # Load the ONE shared GPU model into the main process VRAM (Takes ~2.5GB)
    print("Loading PyTorch models into VRAM...")
    ctx = setup(device='cuda')
    print("Models loaded successfully.")

    results = []
    
    # 1. Start a Pool of CPU Processes to handle SpaCy without GIL
    with ProcessPoolExecutor(max_workers=num_cpu_workers, mp_context=mp.get_context('spawn'), initializer=init_spacy_worker) as cpu_pool:
        # 2. Start a Pool of Threads to handle GPU PyTorch without cloning VRAM
        with ThreadPoolExecutor(max_workers=num_gpu_threads) as gpu_pool:
            
            # Submit all SpaCy tasks
            spacy_futures = {cpu_pool.submit(cpu_process_spacy, file_name, text): file_name for file_name, text in tasks}
            
            # As soon as a SpaCy task finishes, pass it instantly to a GPU thread
            gpu_futures = {}
            for future in tqdm(as_completed(spacy_futures), total=len(tasks), desc="SpaCy CPU Parsing"):
                file_name, text_content, tokens_df, error = future.result()
                
                if error:
                    results.append((file_name, 'error_spacy', error))
                else:
                    g_fut = gpu_pool.submit(gpu_thread_task, file_name, text_content, tokens_df, str(root), ctx)
                    gpu_futures[g_fut] = file_name

            # Wait for all GPU tasks to finish
            for future in tqdm(as_completed(gpu_futures), total=len(gpu_futures), desc="PyTorch GPU Inference"):
                results.append(future.result())

    return pd.DataFrame(results, columns=['file_name', 'status', 'error'])

if __name__ == '__main__':
    df = pd.read_csv("~/medialab/lacroix.csv", low_memory=False, nrows=500)
    df['url_id'] = df['url'].apply(lambda u: re.sub(r'[^\w\-]+', '_', str(u)).strip('_'))
    text_columns = ["text"]
    df[text_columns] = df[text_columns].fillna('').astype(str)

    # We can crank up the concurrency! 
    # VRAM will remain very low (just 1 model + activations), while CPUs and GPU will hit 100%
    status_df = process_dataframe_advanced(
        df,
        text_columns=text_columns,
        root_directory='/home/decourson/propp_press/outputs/',
        id_column='url_id',
        num_cpu_workers=32,   # Request 32 CPUs in SLURM
        num_gpu_threads=16    # 16 concurrent streams hitting the GPU model
    )
    
    print(status_df['status'].value_counts())
    status_df.to_csv('/home/decourson/propp_press/outputs/_status.csv', index=False)
