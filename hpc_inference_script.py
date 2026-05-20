from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import os
import re
import torch
import torch.nn as nn

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from propp_fr import (
    generate_tokens_df, get_embedding_tensor_from_tokens_df,
    generate_entities_df, add_features_to_entities, perform_coreference,
    extract_attributes, generate_characters_dict,
    save_entities_df, save_book_file,
    load_tokenizer_and_embedding_model, load_models,
)


def _move_modules(obj, device, seen=None):
    """Recursively move all nn.Module instances inside obj to device."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))

    if isinstance(obj, nn.Module):
        obj.to(device).eval()
        return  # parameters() already covers submodules

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


def _find_modules(obj, prefix='', seen=None, depth=0, max_depth=4):
    """Print device of every nn.Module found inside obj. For sanity check."""
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > max_depth:
        return
    seen.add(id(obj))

    if isinstance(obj, nn.Module):
        try:
            device = next(obj.parameters()).device
            print(f"  {prefix}  [{type(obj).__name__}]: {device}")
        except StopIteration:
            pass
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            _find_modules(v, f"{prefix}.{k}", seen, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _find_modules(v, f"{prefix}[{i}]", seen, depth + 1, max_depth)
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            if k.startswith('_'):
                continue
            _find_modules(v, f"{prefix}.{k}", seen, depth + 1, max_depth)


def setup(device='cuda', verify=True):
    """Load models once and move ALL neural components (including wrapped ones) to GPU."""
    assert torch.cuda.is_available(), "CUDA not available — check drivers / torch install"
    torch.set_num_threads(os.cpu_count() or 8)

    # Perf knobs — free wins on Ampere/Ada
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    spacy_model, mentions_detection_model, coreference_resolution_model = load_models()
    tokenizer, embedding_model = load_tokenizer_and_embedding_model(
        mentions_detection_model['base_model_name']
    )

    # Move everything to GPU, including wrapped modules inside dicts/objects.
    embedding_model = embedding_model.to(device).eval()
    _move_modules(mentions_detection_model, device)
    _move_modules(coreference_resolution_model, device)

    if verify:
        print("\n=== Device placement after setup() ===")
        print(f"embedding_model: {next(embedding_model.parameters()).device}")
        print("mentions_detection_model:")
        _find_modules(mentions_detection_model, 'mentions_detection_model')
        print("coreference_resolution_model:")
        _find_modules(coreference_resolution_model, 'coreference_resolution_model')
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")

    return {
        'spacy_model': spacy_model,
        'tokenizer': tokenizer,
        'embedding_model': embedding_model,
        'mentions_detection_model': mentions_detection_model,
        'coreference_resolution_model': coreference_resolution_model,
        'device': device,
    }


def process_one(file_name, text_content, root_directory, ctx):
    """Run the full propp_fr pipeline on a single document."""
    try:
        # 1. Pass the max characters batch (500k chunks)
        tokens_df = generate_tokens_df(
            text_content, 
            ctx['spacy_model'],
            max_char_sentence_length=500000 
        )
        
        with torch.inference_mode():
            # 2. Pass the embedding mini batch (scaled for RTX 6000 Ada)
            tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(
                text_content, 
                tokens_df, 
                ctx['tokenizer'], 
                ctx['embedding_model'],
                mini_batch_size=128
            )
            
            # 3. Pass the mentions detection batch (scaled for RTX 6000 Ada)
            entities_df = generate_entities_df(
                tokens_df, 
                tokens_embedding_tensor, 
                ctx['mentions_detection_model'],
                batch_size=256
            )
            
            entities_df = add_features_to_entities(entities_df, tokens_df)
            
            # 4. Coreference batch size (large by default, explicit here for safety)
            entities_df = perform_coreference(
                entities_df, 
                tokens_embedding_tensor, 
                ctx['coreference_resolution_model'],
                batch_size=50000,
                propagate_coref=True, # Added matching default from main package
                rule_based_postprocess=False # Added matching default from main package
            )
            
        tokens_df = extract_attributes(entities_df, tokens_df)
        characters_dict = generate_characters_dict(tokens_df, entities_df)
        save_entities_df(entities_df, file_name, root_directory)
        save_book_file(characters_dict, file_name, root_directory)
        return (file_name, 'ok', None)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return (file_name, 'oom', f'{type(e).__name__}: {e}')
    except Exception as e:
        return (file_name, 'error', f'{type(e).__name__}: {e}')


def process_dataframe(df, text_columns, root_directory, id_column, separator='\n\n'):
    root = Path(root_directory)
    root.mkdir(parents=True, exist_ok=True)

    ctx = setup(device='cuda')

    tasks = [
        (
            str(getattr(row, id_column)),
            separator.join(getattr(row, c) for c in text_columns),
        )
        for row in df.itertuples(index=False)
    ]

    results = []
    for file_name, text_content in tqdm(tasks, desc='propp_fr (GPU)'):
        results.append(process_one(file_name, text_content, str(root), ctx))

    return pd.DataFrame(results, columns=['file_name', 'status', 'error'])


if __name__ == '__main__':
    df = pd.read_csv("~/medialab/lacroix.csv", low_memory=False,nrows=500)
    df['url_id'] = df['url'].apply(lambda u: re.sub(r'[^\w\-]+', '_', str(u)).strip('_'))
    text_columns = ["text"]
    df[text_columns] = df[text_columns].fillna('').astype(str)

    # Suggestion: test on a sample first to time things and confirm everything works
    # df = df.head(200)

    status_df = process_dataframe(
        df,
        text_columns=text_columns,
        root_directory='/home/decourson/propp_press/outputs/',
        id_column='url_id',
    )
    print(status_df['status'].value_counts())
    status_df.to_csv('/home/decourson/propp_press/outputs/_status.csv', index=False)
