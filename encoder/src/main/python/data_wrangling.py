#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from tqdm.notebook import tqdm

from configuration import config
from transformers import AutoTokenizer

# enable tqdm in pandas
# tqdm.pandas()

# map labels to the first token in each word
def align_labels(word_ids: list[int], labels: list[str], label_to_index: dict[str, int]) -> list[int]:
    label_ids = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word_id:
            # ignore if not a word or word id has already been seen
            label_ids.append(config.ignore_index)
        else:
            # get label id for corresponding word
            label_id = label_to_index[labels[word_id]]
            label_ids.append(label_id)
        # remember this word id
        previous_word_id = word_id
    
    return label_ids
            
# build a set of labels in the dataset
def read_label_set(fn: str) -> list[str]:
    labels = set()
    with open(fn) as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                label = tokens[-1]
                labels.add(label)
    return labels

# converts a two-column file in the basic MTL format ("word \t label") into a dataframe
def read_dataframe(fn: str, label_to_index: dict[str, int], task_id: int, tokenizer: AutoTokenizer):
    # now build the actual dataframe for this dataset
    data = {'words': [], 'str_labels': [], 'input_ids': [], 'word_ids': [], 'labels': [], 'task_ids': []}
    with open(fn) as f:
        sent_words = []
        sent_labels = [] 
        for _, line in tqdm(enumerate(f)):
            tokens = line.strip().split()
            if not tokens:
                data['words'].append(sent_words)
                data['str_labels'].append(sent_labels)
                
                # tokenize each sentence
                token_input = tokenizer(sent_words, is_split_into_words = True)  
                token_ids = token_input['input_ids']
                word_ids = token_input.word_ids(batch_index = 0)
                
                # map labels to the first token in each word
                token_labels = align_labels(word_ids, sent_labels, label_to_index)
                
                data['input_ids'].append(token_ids)
                data['word_ids'].append(word_ids)
                data['labels'].append(token_labels)
                data['task_ids'].append(task_id)
                sent_words = []
                sent_labels = [] 
            else:
                sent_words.append(tokens[0])
                sent_labels.append(tokens[1])
    result = pd.DataFrame(data) # TODO: what type is this?
    return pd.DataFrame(data)
