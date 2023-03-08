#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random

from tqdm.notebook import tqdm

from configuration import device, seed, ignore_index, HEAD_POSITIONS

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
            if labels[word_id] in label_to_index.keys():
              label_id = label_to_index[labels[word_id]]
              label_ids.append(label_id)
            else:
              raise Exception(f'Can not find index for label {labels[word_id]}!')
        # remember this word id
        previous_word_id = word_id
    
    return label_ids

# map word-level head positions to subword tokens
def align_head_positions(word_ids, word_heads):
  # map from word positions to first-in-word token positions 
  #print(f'word_ids = {word_ids}')
  #print(f'word_heads = {word_heads}')
  word_to_token_map = {}
  previous_word_id = None
  for i in range(0, len(word_ids)):
    # we are seeing the first token in a word
    if(word_ids[i] != None and word_ids[i] != previous_word_id):
      word_to_token_map[word_ids[i]] = i
    previous_word_id = word_ids[i]

  #print(f'word_to_token_map = {word_to_token_map}')

  # stores the position of the token that is the head for each word
  token_head_positions = []
  previous_word_id = None
  for i in range(0, len(word_ids)):
      word_id = word_ids[i]

      # we are inside an existing word or looking at [CLS]
      if word_id is None or word_id == previous_word_id:
          # the position of this head does not matter since it's not used in the loss
          token_head_positions.append(0)
      # beginning of a word whose head is root (-1)
      elif word_heads[word_id] == -1: 
          # we append the current position here
          # this means that in dual mode we concatenate the same embedding to itself
          token_head_positions.append(i)
      # beginning of a word whose head is not root
      else:
          # get head position for corresponding word
          token_head_positions.append(word_to_token_map[word_heads[word_id]])
      # remember this word id
      previous_word_id = word_id

  return token_head_positions

def make_empty_list(length, value):
  l = []
  for i in range(0, length):
    l.append(value)
  return l
            
# build a sorted list of labels in the dataset            
def read_label_set(fn):
    labels = set()
    with open(fn) as f:
        for index, line in enumerate(f):
            line = line.strip()
            tokens = line.split()
            if tokens != []:
                label = tokens[1] # labels are always on the second position
                labels.add(label)
    print(f'label size = {len(labels)}')
    sorted_labels = list(labels)
    sorted_labels.sort()
    print(f"Using labels: {sorted_labels}")
    return sorted_labels

# converts a two-column file in the basic MTL format ("word \t label") into a dataframe
def read_dataframe(fn, label_to_index, task_id, tokenizer):
    # now build the actual dataframe for this dataset
    data = {'words': [], 'str_labels': [], 'input_ids': [], 'word_ids': [], 'labels': [], HEAD_POSITIONS: [], 'task_ids': []}
    with open(fn) as f:
        sent_words = []
        sent_labels = [] 
        head_positions = []

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
                assert len(word_ids) == len(token_ids)
                
                # map labels to the first token in each word
                token_labels = align_labels(word_ids, sent_labels, label_to_index)
                assert len(token_labels) == len(token_ids)

                # if present, map head offsets to the first token in each word
                token_head_positions = align_head_positions(word_ids, head_positions) if len(head_positions) > 0 else None
                if token_head_positions != None:
                  assert len(token_head_positions) == len(token_ids)
                
                data['input_ids'].append(token_ids)
                data['word_ids'].append(word_ids)
                data['labels'].append(token_labels)
                if token_head_positions != None:
                  data[HEAD_POSITIONS].append(token_head_positions)
                else:
                  data[HEAD_POSITIONS].append(make_empty_list(len(word_ids), ignore_index))
                data['task_ids'].append(task_id)

                #if task_id == 4:
                #  print(f'sent_words = {sent_words}')
                #  print(f'input_ids = {token_ids}')
                #  print(f'word_ids = {word_ids}')
                #  print(f'head_positions = {head_positions}')
                #  print(f'token_head_positions = {token_head_positions}')                  

                sent_words = []
                sent_labels = [] 
                head_positions = []
            else:
                sent_words.append(tokens[0]) # tokens 
                sent_labels.append(tokens[1]) # labels
                if(len(tokens) > 2):
                  head_positions.append(int(tokens[2])) # absolute position of head token 
    return pd.DataFrame(data)
