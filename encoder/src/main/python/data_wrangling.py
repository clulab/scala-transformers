#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random

from tqdm.notebook import tqdm

from configuration import device, seed, ignore_index

# enable tqdm in pandas
# tqdm.pandas()

# map labels to the first token in each word
def align_labels(word_ids, labels, label_to_index):
    label_ids = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word_id:
            # ignore if not a word or word id has already been seen
            label_ids.append(ignore_index)
        else:
            # get label id for corresponding word
            label_id = label_to_index[labels[word_id]]
            label_ids.append(label_id)
        # remember this word id
        previous_word_id = word_id
    
    return label_ids
            
# build a set of labels in the dataset            
def read_label_set(fn):
    labels = set()
    with open(fn) as f:
        for index, line in enumerate(f):
            line = line.strip()
            tokens = line.split()
            if tokens != []:
                label = tokens[-1]
                labels.add(label)
    return labels

# converts a two-column file in the basic MTL format ("word \t label") into a dataframe
def read_dataframe(fn, label_to_index, task_id, tokenizer):
    # now build the actual dataframe for this dataset
    data = {'words': [], 'str_labels': [], 'input_ids': [], 'word_ids': [], 'labels': [], 'task_ids': []}
    with open(fn) as f:
        sent_words = []
        sent_labels = [] 
        for index, line in tqdm(enumerate(f)):
            line = line.strip()
            tokens = line.split()
            if tokens == []:
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
    return pd.DataFrame(data)
