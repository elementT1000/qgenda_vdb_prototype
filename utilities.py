from langchain.prompts import PromptTemplate
#from text import split_text
from typing import List, Literal
import numpy as np
import torch
import os
import re

from transformers import (
    GPT2TokenizerFast
)

import fitz  # PyMuPDF

def inspect_pdf(path_to_pdf: str):
    # Open the PDF document
    pdf_document = path_to_pdf  # Replace with your PDF file path
    doc = fitz.open(pdf_document)

    # Iterate through each page
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        print(f"Page {page_num + 1}:")

        # Retrieve font information
        fonts = page.get_fonts(full=True)
        for font in fonts:
            xref, ext, font_type, basefont, name, encoding, emb = font
            print(f"  Font Name: {basefont}")
            print(f"    Type: {font_type}")
            print(f"    Encoding: {encoding}")
            print(f"    Embedded: {'Yes' if emb else 'No'}")
            print(f"    XRef: {xref}")
            print(f"    Font Dictionary Name: {name}")
            print(f"    Extension: {ext}")

def clean_text(text):
    # Replace specific problematic Unicode characters with desired replacements
    text = text.replace('\u2019', "'")  # Replace \u2019 with an apostrophe
    text = text.replace('\u00a9', '©')  # Replace \u00a9 with the copyright symbol
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Replace curly quotes with straight quotes
    text = text.replace('\u2014', '—')  # Replace em dash with proper character
    text = text.replace('\u00c2', '')  # Remove \u00c2, which may be an encoding artifact

    # Use a regex to remove any other unwanted Unicode sequences that may not have been explicitly addressed
    text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)  # Remove any Unicode character represented by \uXXXX

    # Clean the text by removing extra whitespace and other artifacts
    text = re.sub(r'\n+', ' ', text)  # Replace new lines (\n and \n\n) with a space
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces for clean flow
    text = re.sub(r'n\d+%.*?\n', '', text)  # Remove numerical charting text
    
    return text.strip()

def get_openai_key():
    openai_key = os.getenv('OPENAI_API_KEY')

    if openai_key:
        print("OpenAI key accessed.")
    else:
        print("API Key not found. Check your environment variables.")

    return openai_key


def qna_prompt():
    """ 
    prompt template for q and a type answering
    """

    template = """Given the following extracted parts of a long document ("SOURCES") and a question ("QUESTION"), create a final answer one paragraph long. 
    Don't try to make up an answer and use the text in the SOURCES only for the answer. If you don't know the answer, just say that you don't know. 
    QUESTION: {question}
    =========
    SOURCES:
    {summaries}
    =========
    ANSWER:"""
    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
    return PROMPT

def clean_txt_prompt():
    template = """
    Given the following extracted parts of a PDF document ("SOURCES"), transcribe only the primary text of the document while strictly removing unnecessary symbols, formatting, and metadata. Please follow these guidelines:

    1. **Remove Encoding Symbols and Format the Text for Natural Flow**:
    - **Eliminate all newline symbols (`\n`, `\n\n`) completely**: Remove newline symbols even if they are embedded in the text or touching words. Ensure that sentences flow naturally, replacing these symbols with a space where appropriate.
    - **Replace Unicode characters**: Replace symbols like `\u2019` with their corresponding character (e.g., `\u2019` should become `'`).

    2. **Ignore Non-Primary or Unnecessary Content**:
    - Skip any metadata or text that is not part of the main narrative (e.g., "Copyright © 2024").
    - Skip numerical sequences and charting text like `"n40%\n0%\n30%\n20%\n10%"`, and any similar chart or graph label sequences.

    3. **Transcription Expectations**:
    - Do **not** add any boilerplate or explanatory text.
    - Ensure that the transcription is coherent, preserving only the meaningful portions of the document, with all sentences flowing smoothly.

    **Note**: The final transcription should be a **clean, readable version** of the main content, without any formatting symbols, numerical sequences, metadata, or extraneous information.

    SOURCES:
    {summaries}
    =========
    ANSWER:

    """
    prompt = PromptTemplate(template=template, input_variables=["summaries"])

    return prompt


'''def predict_ce(query, texts, model=None, key='ce'):
    """ 
    score all the queries with respect to the texts
    """

    if model is None:
        if key not in model_cache:
            model, tokenizer = load_ce()
            model_cache[key] = (model, tokenizer) 
        else:
            (model, tokenizer) = model_cache[key]

    # create pairs
    softmax = torch.nn.Softmax(dim=0)
    N = len(texts)
    queries = [query]*N
    pairs = list(zip(queries, texts))
    features = tokenizer(pairs,  padding=True, truncation=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    return softmax(scores)'''

def get_sorted_inds(scores):
    """ 
    return indexes based on sorted scores
    """
    return scores.argsort(0, descending=True)

def extract_text_from_highlights(res, token_limit=256, truncate=True):

    highlights = []
    texts = []
    for ind,hit in enumerate(res[ResultsFields.hits]):
        highlight_list = hit[ResultsFields.highlights]
        highlight_key = list(highlight_list[0].keys())[0]
        highlight_text = list(highlight_list[0].values())[0]
        text = hit[highlight_key]
    
        if truncate:
            text = " ".join(text.split())
            highlight_text = " ".join(highlight_text.split())
            text = truncate_text(text, token_limit, highlight_text)
            
        texts.append(text)
        highlights.append(highlight_text)

    return highlights, texts


class ResultsFields:
    hits = 'hits'
    highlights = '_highlights'

cached_tokenizers = dict()

def _lies_between(offset_tuple, offset):
    """ 
    given a tuple of ints, determine if offset lies between them
    """
    return offset >= offset_tuple[0] and offset < offset_tuple[1]


def _find_end_character_mapping(offset_mapping, offset):
    """assumes sorted offset_mapping. unless this was modified 
       this will be the default from the tokenizer
    """
    # if the max length is bigger we just return the last index
    if offset >= max(offset_mapping[-1]):
        return [offset_mapping[-1]]
    return [ind for ind in offset_mapping if _lies_between(ind, offset)]

def find_highlight_index_in_text(text, highlight):
    """ 
    return start and end character indices for the sub-string (highlight)
    """
    if highlight not in text:
        return (None, None)

    # returns left right
    left_ind = text.index(highlight)
    right_ind = left_ind + len(highlight)

    return (left_ind, right_ind)


def get_token_indices(text: str, token_limit: int,  
                method: Literal['start', 'end', 'center','offset'] = 'start',
                tokenizer = None,
                offset: int = None):

    # leave it here instead of a paramter
    default_tokenizer = 'gpt2'

    if tokenizer is None:
        if default_tokenizer not in cached_tokenizers:
            tokenizer = GPT2TokenizerFast.from_pretrained(default_tokenizer)
            cached_tokenizers[default_tokenizer] = tokenizer
        else:
            tokenizer = cached_tokenizers[default_tokenizer]

    tokenized_text = tokenizer(text, return_offsets_mapping=True)
    token_ids = tokenized_text['input_ids']
    character_offsets = tokenized_text['offset_mapping']
    text_token_len = len(token_ids)

    # need to get the offset from the start to hit the full size
    delta = text_token_len - token_limit

    # nothing to do if it fits already
    if delta <= 0:
        return [character_offsets[0], character_offsets[-1]]

    # convert offset into token space
    character_offset_tuple = _find_end_character_mapping(character_offsets, offset)
    token_offset = character_offsets.index(character_offset_tuple[0])

    is_odd_offset = 1
    if token_limit % 2 == 1: is_odd_offset = 0

    if method == 'start':
        ind_start = character_offsets[0]
        ind_end = character_offsets[token_limit-1]

    elif method == 'end':
        ind_start = character_offsets[delta]
        ind_end = character_offsets[-1]

    elif method == 'center':       
        center_token = text_token_len//2
        left_ind = max(center_token - token_limit//2, 0)
        right_ind = min(center_token + token_limit//2, text_token_len)
        ind_start = character_offsets[left_ind]
        ind_end = character_offsets[right_ind-is_odd_offset]
    
    elif method == 'offset':
        center_token = token_offset 
        left_ind = max(center_token - token_limit//2, 0)
        right_ind = min(center_token + token_limit//2, text_token_len)
        ind_start = character_offsets[left_ind]
        ind_end = character_offsets[right_ind-is_odd_offset]
    
    else: 
        raise RuntimeError("incorrect method specified")

    return ind_start, ind_end

def truncate_text(text, token_limit, highlight=None):
    """ 
    truncates text to a token limit centered on the highlight text
    """

    if highlight is None:
        method = 'start'
        center_ind = 0 # this will not be used for this start method
    else:
        # TODO the full context may ot get used if the highlight is not centered    
        # we would need to add the excess to the end/start

        method = 'offset'
        # get indices of highlight
        inds = find_highlight_index_in_text(text, highlight)
        # get the center of the highlight in chars
        center_ind = (max(inds) - min(inds))//2 + min(inds)
        # now map this to tokens and get the left/right char indices to achieve token limit

    ind_left, ind_right = get_token_indices(text, token_limit, method=method, offset=center_ind)
    trunc_text = text[min(ind_left):max(ind_right)]

    return trunc_text

def check_highlights_field(hit, highlight=ResultsFields.highlights):
    """
    check the validity of the highlights in the hit
    """

    if highlight in hit:
        if len(hit[highlight]) == 0:
            return False
        elif isinstance(hit[highlight], dict):
            if hit[highlight].values() == 0:
                return False
            else:
                return True
        else:
            raise RuntimeError("invalid hits and highlights")
    else:
        return False


def test_truncate():
    import random
    import string
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    text_en = ['hello this is a test sentence. this is another one? i think so. maybe throw in some more letteryz....xo']
    text_rm = [''.join(random.choices(string.ascii_uppercase + string.digits, k=10000)) for _ in range(10)]
    ks = [1, 32, 128, 1024, 2048]
    texts = text_en + text_rm
    methods = ['offset', 'start', 'end', 'center']
    for text in texts:
        k_gt = len(tokenizer(text)['input_ids'])
        for k in ks:
            for method in methods:
                ind_left, ind_right = get_token_indices(text, k, method=method, offset=2)
                trunc_text = text[min(ind_left):max(ind_right)]
                k_fn = len(tokenizer(trunc_text)['input_ids'])
                
                assert k_fn <= min(k,k_gt)


def test_find_highlight_in_text():

    n_highlights = 5
    
    texts = [
        'hello how are you', 
        "I assume you only want to find the first occurrence of word in phrase. If that's the case, just use str.index to get the position of the first character. Then, add len(word) - 1 to it to get the position of the last character.",
        ]

    for text in texts:
        for _ in range(n_highlights):
            highlight_ind = sorted(np.random.choice(list(range(len(text))),2, replace=False))
            highlight = text[highlight_ind[0]:highlight_ind[1]]
            inds = find_highlight_index_in_text(text, highlight)
            assert text[inds[0]:inds[1]] == highlight


