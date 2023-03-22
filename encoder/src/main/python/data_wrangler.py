#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from file_utils import FileUtils
from names import names
from parameters import parameters
from tqdm.notebook import tqdm
from transformers import AutoTokenizer

# enable tqdm in pandas
# tqdm.pandas()

class DataWrangler:
    # map labels to the first token in each word
    @classmethod
    def align_labels(cls, word_ids: list[int], labels: list[str], label_to_index: dict[str, int]) -> list[int]:
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None or word_id == previous_word_id:
                # ignore if not a word or word id has already been seen
                label_ids.append(parameters.ignore_index)
            else:
                # get label id for corresponding word
                label_id = label_to_index.get(labels[word_id])
                if label_id is not None:
                    label_ids.append(label_id)
                else:
                    raise Exception(f"Can not find index for label {labels[word_id]}!")
            # remember this word id
            previous_word_id = word_id
        
        return label_ids

    # map word-level head positions to subword tokens
    @classmethod
    def align_head_positions(cls, word_ids: list[int], word_heads: list[int]) -> list[int]:
        # map from word positions to first-in-word token positions 
        #print(f"word_ids = {word_ids}")
        #print(f"word_heads = {word_heads}")
        word_to_token_map = {}
        previous_word_id = None
        for i in range(0, len(word_ids)):
            # we are seeing the first token in a word
            if word_ids[i] != None and word_ids[i] != previous_word_id:
                word_to_token_map[word_ids[i]] = i
            previous_word_id = word_ids[i]

        #print(f"word_to_token_map = {word_to_token_map}")

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
                
    # build a sorted list of labels in the dataset            
    @classmethod
    def read_label_set(cls, filename: str) -> list[str]:
        labels = set()
        with FileUtils.for_reading(filename) as file:
            for line in file:
                tokens = line.strip().split()
                if tokens:
                    label = tokens[1] # labels are always on the second position
                    labels.add(label)
        print(f"label size = {len(labels)}")
        sorted_labels = list(labels)
        sorted_labels.sort()
        print(f"Using labels: {sorted_labels}")
        return sorted_labels

    # converts a two-column file in the basic MTL format ("word \t label") into a dataframe
    @classmethod
    def read_dataframe(cls, filename: str, label_to_index: dict[str, int], task_id: int, tokenizer: AutoTokenizer) -> pd.DataFrame:
        # now build the actual dataframe for this dataset
        data = {"words": [], "str_labels": [], names.INPUT_IDS: [], "word_ids": [], "labels": [], names.HEAD_POSITIONS: [], "task_ids": []}
        with FileUtils.for_reading(filename) as file:
            sent_words = []
            sent_labels = [] 
            head_positions = []

            for line in tqdm(file):
                tokens = line.strip().split()
                if tokens:
                    sent_words.append(tokens[0]) # tokens 
                    sent_labels.append(tokens[1]) # labels
                    if len(tokens) > 2:
                        head_positions.append(int(tokens[2])) # absolute position of head token 
                else:
                    data["words"].append(sent_words)
                    data["str_labels"].append(sent_labels)
                    
                    # tokenize each sentence
                    token_input = tokenizer(sent_words, is_split_into_words=True)  
                    token_ids = token_input[names.INPUT_IDS]
                    word_ids = token_input.word_ids(batch_index=0)
                    assert len(word_ids) == len(token_ids)
                    
                    # map labels to the first token in each word
                    token_labels = DataWrangler.align_labels(word_ids, sent_labels, label_to_index)
                    assert len(token_labels) == len(token_ids)

                    # if present, map head offsets to the first token in each word
                    token_head_positions = DataWrangler.align_head_positions(word_ids, head_positions) if len(head_positions) > 0 else None
                    if token_head_positions != None:
                        assert len(token_head_positions) == len(token_ids)
                    
                    data[names.INPUT_IDS].append(token_ids)
                    data["word_ids"].append(word_ids)
                    data["labels"].append(token_labels)
                    if token_head_positions != None:
                        data[names.HEAD_POSITIONS].append(token_head_positions)
                    else:
                        data[names.HEAD_POSITIONS].append([parameters.ignore_index] * len(word_ids))
                    data["task_ids"].append(task_id)

                    #if task_id == 4:
                    #  print(f"sent_words = {sent_words}")
                    #  print(f"input_ids = {token_ids}")
                    #  print(f"word_ids = {word_ids}")
                    #  print(f"head_positions = {head_positions}")
                    #  print(f"token_head_positions = {token_head_positions}")                  

                    sent_words = []
                    sent_labels = [] 
                    head_positions = []

        return pd.DataFrame(data)
