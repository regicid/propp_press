import subprocess
import sys
from tqdm.auto import tqdm
import gc
import pandas as pd
import torch
import spacy


## Get tokens_df from text_file
def load_spacy_model(model_name='fr_dep_news_trf', model_max_length=500000):

    # Loading Spacy French Transformer Model and Enabling GPU
    spacy.prefer_gpu()

    try:
        model = spacy.load(model_name)
    except OSError:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', model_name], check=True)
        model = spacy.load(model_name)

    model.max_length = model_max_length
    print(f'Loaded Spacy Model: {model_name}')

    # Ensure that the model's transformer is using the GPU if available
    if torch.cuda.is_available():
        print("CUDA is available, model should run on GPU.")
    else:
        print("CUDA is not available, model will run on CPU.")

    return model

def generate_tokens_df_from_spacy_doc(doc):
    token_dict = []
    paragraph_ID = 0
    sentence_ID = -1
    previous_is_newline_char = True
    previous_is_sent_start = False
    previous_is_punct = False
    token_ID_within_sentence = 0

    for token in doc:
        word = token.text
        lemma = token.lemma_
        token_ID_within_document = token.i
        byte_onset = token.idx
        byte_offset = token.idx + len(word)
        is_newline_character = '\n' in token.text_with_ws
        POS_tag = token.pos_
        morph = token.morph
        is_sent_start = token.is_sent_start
        if previous_is_newline_char or ((
                                                token.is_title or token.is_punct) and is_sent_start and previous_is_punct and not previous_is_sent_start):
            is_sent_start = True
        else:
            is_sent_start = False
        dependency_relation = token.dep_
        syntactic_head_ID = token.head.i

        if word in ['.', '!', '?']:
            previous_is_punct = True
        else:
            previous_is_punct = False

        if is_sent_start:
            previous_is_sent_start = True
            sentence_ID += 1
            token_ID_within_sentence = 0
        else:
            previous_is_sent_start = False
            token_ID_within_sentence += 1

        if is_newline_character == True:
            paragraph_ID += 1
            previous_is_newline_char = True
        else:
            previous_is_newline_char = False
            token_dict.append({'paragraph_ID': paragraph_ID,
                               'sentence_ID': sentence_ID,
                               'token_ID_within_sentence': token_ID_within_sentence,
                               'token_ID_within_document': token_ID_within_document,
                               'word': word,
                               'lemma': lemma,
                               'byte_onset': byte_onset,
                               'byte_offset': byte_offset,
                               'POS_tag': POS_tag,
                               'dependency_relation': dependency_relation,
                               'syntactic_head_ID': syntactic_head_ID,
                               'morph': morph,
                               })

    tokens_df = pd.DataFrame(token_dict)
    del doc
    gc.collect()
    torch.cuda.empty_cache()

    token_id_mapping = {}
    for i, previous_token_id in enumerate(tokens_df['token_ID_within_document'].tolist()):
        token_id_mapping[previous_token_id] = i
    # Replace IDs in 'token_ID_within_document' column
    tokens_df['token_ID_within_document'] = tokens_df['token_ID_within_document'].map(token_id_mapping).fillna(
        tokens_df['token_ID_within_document']).astype(int)
    # Replace IDs in 'syntactic_head_ID' column
    tokens_df['syntactic_head_ID'] = tokens_df['syntactic_head_ID'].map(token_id_mapping).fillna(
        tokens_df['syntactic_head_ID']).astype(int)

    return tokens_df

def generate_tokens_df(text_content, spacy_model, max_char_sentence_length=100000, verbose=1):
    text_len = len(text_content)
    sample_count = (text_len // max_char_sentence_length) + 1
    sample_boundaries = [i for i in range(0, text_len, text_len // sample_count)] + [text_len]

    tokens_df = pd.DataFrame()
    start_boundary = 0
    for end_boundary in tqdm(sample_boundaries[1:], desc='Batch Spacy Tokenization', leave=False, disable=(verbose == 0)):
        sample_text = text_content[start_boundary:end_boundary]
        sample_doc = spacy_model(sample_text)
        sample_tokens_df = generate_tokens_df_from_spacy_doc(sample_doc)

        if len(tokens_df) == 0:
            max_sentence_id = sample_tokens_df['sentence_ID'].max()
            last_sentence_start_row = sample_tokens_df[sample_tokens_df['sentence_ID'] == max_sentence_id].iloc[0]
            if end_boundary != sample_boundaries[-1]:
                sample_tokens_df = sample_tokens_df.iloc[:last_sentence_start_row.name]
            tokens_df = sample_tokens_df

        else:
            sample_tokens_df['paragraph_ID'] = sample_tokens_df['paragraph_ID'] + previous_last_sentence_start_row[
                'paragraph_ID']
            sample_tokens_df['sentence_ID'] = sample_tokens_df['sentence_ID'] + previous_last_sentence_start_row[
                'sentence_ID']
            sample_tokens_df['token_ID_within_document'] = sample_tokens_df['token_ID_within_document'] + \
                                                           previous_last_sentence_start_row['token_ID_within_document']
            sample_tokens_df['byte_onset'] = sample_tokens_df['byte_onset'] + previous_last_sentence_start_row[
                'byte_onset']
            sample_tokens_df['byte_offset'] = sample_tokens_df['byte_offset'] + previous_last_sentence_start_row[
                'byte_onset']
            sample_tokens_df['syntactic_head_ID'] = sample_tokens_df['syntactic_head_ID'] + \
                                                    previous_last_sentence_start_row['token_ID_within_document']

            max_sentence_id = sample_tokens_df['sentence_ID'].max()
            last_sentence_start_row = sample_tokens_df[sample_tokens_df['sentence_ID'] == max_sentence_id].iloc[0]
            if end_boundary != sample_boundaries[-1]:
                sample_tokens_df = sample_tokens_df.iloc[:last_sentence_start_row.name]

            tokens_df = pd.concat([tokens_df, sample_tokens_df], ignore_index=True)

        previous_last_sentence_start_row = last_sentence_start_row
        start_boundary = previous_last_sentence_start_row['byte_onset']

        del sample_doc
        gc.collect()
        torch.cuda.empty_cache()

    return tokens_df


import numpy as np
import pandas as pd


def generate_tokens_df_from_spacy_doc_vectorized(doc):
    """
    Version optimisée de generate_tokens_df_from_spacy_doc.
    - Aucune libération mémoire (gc.collect / torch.cuda.empty_cache).
    - Une seule passe d'extraction sur le doc spaCy.
    - paragraph_ID / sentence_ID / token_ID_within_sentence calculés en numpy.
    - Seule reste une petite boucle Python sur des listes pour la récurrence
      sur is_sent_start (dépend de la valeur finale du token précédent).
    """
    n = len(doc)
    columns = ['paragraph_ID', 'sentence_ID', 'token_ID_within_sentence',
               'token_ID_within_document', 'word', 'lemma',
               'byte_onset', 'byte_offset', 'POS_tag',
               'dependency_relation', 'syntactic_head_ID', 'morph']
    if n == 0:
        return pd.DataFrame(columns=columns)

    # ---- 1. Extraction d'attributs en une seule passe ----
    words           = [None] * n
    lemmas          = [None] * n
    pos_tags        = [None] * n
    morphs          = [None] * n
    dep_rels        = [None] * n
    token_ids       = np.empty(n, dtype=np.int64)
    byte_onsets     = np.empty(n, dtype=np.int64)
    byte_offsets    = np.empty(n, dtype=np.int64)
    head_ids        = np.empty(n, dtype=np.int64)
    is_newline      = np.empty(n, dtype=bool)
    is_title        = np.empty(n, dtype=bool)
    is_punct_spacy  = np.empty(n, dtype=bool)
    is_sent_start_s = np.empty(n, dtype=bool)

    for i, t in enumerate(doc):
        text = t.text
        words[i]           = text
        lemmas[i]          = t.lemma_
        pos_tags[i]        = t.pos_
        morphs[i]          = t.morph
        dep_rels[i]        = t.dep_
        token_ids[i]       = t.i
        byte_onsets[i]     = t.idx
        byte_offsets[i]    = t.idx + len(text)
        head_ids[i]        = t.head.i
        is_newline[i]      = '\n' in t.text_with_ws
        is_title[i]        = t.is_title
        is_punct_spacy[i]  = t.is_punct
        is_sent_start_s[i] = bool(t.is_sent_start)

    # ---- 2. Masques booléens vectorisés ----
    is_punct_terminal = np.fromiter(
        (w in ('.', '!', '?') for w in words), dtype=bool, count=n
    )

    # tableaux "précédents" via shift
    prev_newline = np.empty(n, dtype=bool)
    prev_newline[0]  = True              # = previous_is_newline_char initial
    prev_newline[1:] = is_newline[:-1]

    prev_punct = np.empty(n, dtype=bool)
    prev_punct[0]  = False               # = previous_is_punct initial
    prev_punct[1:] = is_punct_terminal[:-1]

    # composante non-récursive de la condition d'override
    # cond_b[i] = (is_title[i] or is_punct[i]) and is_sent_start_spacy[i] and prev_punct[i]
    cond_b = (is_title | is_punct_spacy) & is_sent_start_s & prev_punct

    # ---- 3. Récurrence : f[i] = prev_newline[i] OR (cond_b[i] AND NOT f[i-1]) ----
    pn  = prev_newline.tolist()
    cb  = cond_b.tolist()
    fss = [False] * n
    prev_f = False
    for i in range(n):
        if pn[i] or (cb[i] and not prev_f):
            fss[i] = True
            prev_f = True
        else:
            prev_f = False
    final_is_sent_start = np.asarray(fss, dtype=bool)

    # ---- 4. paragraph_ID / sentence_ID / token_ID_within_sentence vectorisés ----
    # paragraph_ID = nombre de newlines dans [0, i)
    paragraph_ids = np.zeros(n, dtype=np.int64)
    if n > 1:
        np.cumsum(is_newline[:-1], out=paragraph_ids[1:])

    # sentence_ID = cumsum(sent_start) - 1 (l'original démarre à -1)
    sentence_ids = np.cumsum(final_is_sent_start) - 1

    # token_ID_within_sentence = i - dernier indice de sent_start <= i
    indices = np.arange(n, dtype=np.int64)
    last_sent_start = np.where(final_is_sent_start, indices, -1)
    last_sent_start = np.maximum.accumulate(last_sent_start)
    token_ids_within_sent = indices - last_sent_start

    # ---- 5. Filtrage des tokens newline ----
    mask = ~is_newline
    keep_idx = np.where(mask)[0].tolist()

    # remap token_ID_within_document -> 0..k-1
    kept_token_ids = token_ids[mask]
    new_token_ids  = np.arange(len(kept_token_ids), dtype=np.int64)
    id_mapping = dict(zip(kept_token_ids.tolist(), new_token_ids.tolist()))

    kept_head_ids = head_ids[mask]
    new_head_ids = np.fromiter(
        (id_mapping.get(int(h), int(h)) for h in kept_head_ids),
        dtype=np.int64, count=len(kept_head_ids)
    )

    # ---- 6. DataFrame final ----
    return pd.DataFrame({
        'paragraph_ID':              paragraph_ids[mask],
        'sentence_ID':               sentence_ids[mask],
        'token_ID_within_sentence':  token_ids_within_sent[mask],
        'token_ID_within_document':  new_token_ids,
        'word':                      [words[i]    for i in keep_idx],
        'lemma':                     [lemmas[i]   for i in keep_idx],
        'byte_onset':                byte_onsets[mask],
        'byte_offset':               byte_offsets[mask],
        'POS_tag':                   [pos_tags[i] for i in keep_idx],
        'dependency_relation':       [dep_rels[i] for i in keep_idx],
        'syntactic_head_ID':         new_head_ids,
        'morph':                     [morphs[i]   for i in keep_idx],
    })


def generate_tokens_df_small(text_content, spacy_model):
    """
    Variante 'petits documents' : pas de batching, pas de gc/cache,
    boucle de tokens vectorisée.
    """
    doc = spacy_model(text_content)
    return generate_tokens_df_from_spacy_doc_vectorized(doc)
