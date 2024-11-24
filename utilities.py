import glob
import pandas as pd
from langchain.prompts import PromptTemplate
import re
#from text import split_text
from typing import List, Literal
import numpy as np
import torch
import os

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    GPT2TokenizerFast
)

def get_openai_key():
    openai_key = os.getenv('OPENAI_API_KEY')

    if openai_key:
        print("OpenAI key accessed.")
    else:
        print("API Key not found. Check your environment variables.")

    return openai_key

def reformat_npcs(npcs):
    """ 
    helper function to make the data tidy
    """
    new_npcs = []
    for npc in npcs:

        # include a name field in each for filtering
        name = npc['name']

        for key,value in npc.items():
            doc = {"name":name, "text":f"my {key} is {value}"}
            new_npcs.append(doc)
    return new_npcs

def marqo_template():
    """
    holds the prompt template
    """
    template = """The following is a conversation with a fictional superhero in a movie. 
    BACKGROUND is provided which describes some the history and powers of the superhero. 
    The conversation should always be consistent with this BACKGROUND. 
    Continue the conversation as the superhero in the movie and **always** use something from the BACKGROUND. 
    You are very funny and talkative and **always** talk about your superhero skills in relation to your BACKGROUND.
    BACKGROUND:
    =========
    {summaries}
    =========
    Conversation:
    {conversation}
    """
    return template

def marqo_prompt(template = marqo_template()):
    """ 
    thin wrapper for prompt creation
    """
    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "conversation"])
    return PROMPT

def read_md_file(filename):
    """ 
    generic md/txt file reader
    """
    with open(filename, 'r') as f:
        return f.read()

def clean_md_text(text):
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove inline code
    text = re.sub(r'`.*?`', '', text)
    
    # Remove headings
    text = re.sub(r'#+.*?\n', '', text)
    
    # Remove horizontal lines
    text = re.sub(r'---*', '', text)
    
    # Remove links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    # Remove emphasis
    text = re.sub(r'\*\*.*?\*\*', '', text)
    text = re.sub(r'\*.*?\*', '', text)
    
    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    return text

'''
def load_all_files(files):
    """ 
    wrapper to load and clean text files
    """

    results = []
    for f in files:
        text = read_md_file(f)
        splitted_text = split_text(text, split_length=10, split_overlap=3)
        cleaned_text = [clean_md_text(_text) for _text in splitted_text]   
        _files = [f]*(len(cleaned_text))

        results += list(zip(_files, splitted_text, cleaned_text))

    return pd.DataFrame(results, columns=['filename', 'text', 'cleaned_text'])
'''
'''def load_data():
    """  
    wrapper to load all the data files
    """
    marqo_docs_directory = 'data/'
    files = glob.glob(marqo_docs_directory + 'p*.txt', recursive=True)
    files = [f for f in files if not f.startswith('_')]
    return load_all_files(files)'''

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

model_cache = dict()
def load_ce(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """ 
    loads the sbert cross-encoder model
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def predict_ce(query, texts, model=None, key='ce'):
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

    return softmax(scores)

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


