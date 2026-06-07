from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import os
import re
import torch
import torch.nn as nn
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from propp_fr import (
    generate_tokens_df, get_embedding_tensor_from_tokens_df,
    generate_entities_df, add_features_to_entities, perform_coreference,
    extract_attributes, generate_characters_dict,
    save_entities_df, save_book_file, save_tokens_df,
    load_tokenizer_and_embedding_model, load_models,
    load_ontology_classification_model, classify_attributes,
)

# Global context for each worker process
worker_ctx = None

def _move_modules(obj, device, dtype=None, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))

    if isinstance(obj, nn.Module):
        if dtype:
            obj.to(device=device, dtype=dtype).eval()
        else:
            obj.to(device).eval()
        return

    if isinstance(obj, dict):
        for v in obj.values():
            _move_modules(v, device, dtype, seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _move_modules(v, device, dtype, seen)
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            if not k.startswith('_'):
                _move_modules(v, device, dtype, seen)

def setup(device='cuda'):
    assert torch.cuda.is_available(), "CUDA not available"
    torch.set_num_threads(1)  # Important: limit CPU threads per worker to avoid oversubscription

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    spacy_model, mentions_detection_model, coreference_resolution_model = load_models()
    tokenizer, embedding_model = load_tokenizer_and_embedding_model(
        mentions_detection_model['base_model_name']
    )

    dtype = torch.bfloat16

    embedding_model = embedding_model.to(device=device, dtype=dtype).eval()
    _move_modules(mentions_detection_model, device, dtype)
    _move_modules(coreference_resolution_model, device, dtype)

    attribute_classification_model = load_ontology_classification_model(
        device=torch.device(device)
    )
    _move_modules(attribute_classification_model, device, dtype)

    return {
        'spacy_model': spacy_model,
        'tokenizer': tokenizer,
        'embedding_model': embedding_model,
        'mentions_detection_model': mentions_detection_model,
        'coreference_resolution_model': coreference_resolution_model,
        'attribute_classification_model': attribute_classification_model,
        'device': device,
    }

def init_worker():
    """Initializes the models on the GPU for each separate process."""
    global worker_ctx
    import time
    import random

    # Stagger worker initialization to prevent massive simultaneous VRAM allocations
    time.sleep(random.uniform(0, 30))

    worker_ctx = setup(device='cuda')

def process_one_task(args):
    """Wrapper to unpack arguments and run process_one with the global worker_ctx."""
    file_name, text_content, root_directory = args
    try:
        # 1. Pass the max characters batch (500k chunks)
        tokens_df = generate_tokens_df(
            text_content,
            worker_ctx['spacy_model'],
            max_char_sentence_length=500000
        )

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 2. Pass the embedding mini batch
            tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(
                text_content,
                tokens_df,
                worker_ctx['tokenizer'],
                worker_ctx['embedding_model'],
                mini_batch_size=128
            )

            # 3. Pass the mentions detection batch
            entities_df = generate_entities_df(
                tokens_df,
                tokens_embedding_tensor,
                worker_ctx['mentions_detection_model'],
                batch_size=256
            )

            entities_df = add_features_to_entities(entities_df, tokens_df)

            # 4. Coreference batch size
            entities_df = perform_coreference(
                entities_df,
                tokens_embedding_tensor,
                worker_ctx['coreference_resolution_model'],
                batch_size=50000,
                propagate_coref=True,
                rule_based_postprocess=False
            )

            tokens_df = extract_attributes(entities_df, tokens_df)

            # Attribute ontology classification (runs on the already-computed
            # embedding tensor — no extra GPU pass needed)
            tokens_df = classify_attributes(
                tokens_df,
                tokens_embedding_tensor,
                worker_ctx['attribute_classification_model'],
            )

        characters_dict = generate_characters_dict(tokens_df, entities_df)
        save_entities_df(entities_df, file_name, root_directory)
        save_book_file(characters_dict, file_name, root_directory)
        return (file_name, 'ok', None)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return (file_name, 'oom', f'{type(e).__name__}: {e}')
    except Exception as e:
        return (file_name, 'error', f'{type(e).__name__}: {e}')

def process_dataframe_parallel(df, text_columns, root_directory, id_column, num_workers=8, separator='\n\n'):
    root = Path(root_directory)
    root.mkdir(parents=True, exist_ok=True)
    root_str = str(root)

    def task_iter():
        for row in df.itertuples(index=False):
            file_name = str(getattr(row, id_column))
            # Skip if already processed (book file is the last thing written)
            if os.path.exists(os.path.join(root_str, f'{file_name}.book')):
                continue
            yield (
                file_name,
                separator.join(getattr(row, c) for c in text_columns),
                root_str,
            )

    results = []
    ctx = mp.get_context('spawn')
    pending = set()
    max_in_flight = num_workers * 2

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx, initializer=init_worker) as executor:
        it = task_iter()

        # Prime the pool
        for _ in range(max_in_flight):
            try:
                pending.add(executor.submit(process_one_task, next(it)))
            except StopIteration:
                break

        already_done = sum(
            1 for row in df.itertuples(index=False)
            if os.path.exists(os.path.join(root_str, f'{str(getattr(row, id_column))}.book'))
        )
        total_todo = len(df) - already_done
        pbar = tqdm(total=total_todo, desc=f'propp_fr (GPU x{num_workers} workers)')

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                results.append(f.result())
                pbar.update(1)
                try:
                    pending.add(executor.submit(process_one_task, next(it)))
                except StopIteration:
                    pass
        pbar.close()

    return pd.DataFrame(results, columns=['file_name', 'status', 'error'])

if __name__ == '__main__':
    text_columns = ["text"]

    root_directory = '/home/decourson/propp_press/outputs/'

    # Only load the columns we actually need
    df = pd.read_csv(
        "~/medialab/lemonde.csv",
        usecols=['url'] + text_columns,
        dtype=str,
    )
    df['url_id'] = df['url'].apply(lambda u: re.sub(r'[^\w\-]+', '_', str(u)).strip('_'))
    df[text_columns] = df[text_columns].fillna('').astype(str)

    # Drop the raw url column; we only need url_id from here on
    df = df[['url_id'] + text_columns]
    print(f'Loaded {len(df)} articles. Already-processed files will be skipped automatically.')

    # Note: 18 workers will use ~54GB of VRAM (~3GB per process) on a 48GB Ada.
    # If OOM occurs, lower num_workers to 12 or 8.
    status_df = process_dataframe_parallel(
        df,
        text_columns=text_columns,
        root_directory=root_directory,
        id_column='url_id',
        num_workers=18
    )

    print(status_df['status'].value_counts())
    status_df.to_csv(os.path.join(root_directory, '_status.csv'), index=False)