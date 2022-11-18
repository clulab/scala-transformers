#!/usr/bin/env python
# coding: utf-8

from data_wrangling import read_label_set, read_dataframe

class Task():
    def __init__(self, task_id, task_name, train_file_name, dev_file_name, test_file_name, tokenizer):
        self.task_id = task_id
        self.task_name = task_name
        # we need an index of labels first
        self.labels = read_label_set(train_file_name)
        self.index_to_label = {i:t for i,t in enumerate(self.labels)}
        self.label_to_index = {t:i for i,t in enumerate(self.labels)}
        self.num_labels = len(self.index_to_label)
        # create data frames for the datasets
        self.train_df = read_dataframe(train_file_name, self.label_to_index, self.task_id, tokenizer)
        self.dev_df = read_dataframe(dev_file_name, self.label_to_index, self.task_id, tokenizer)
        self.test_df = read_dataframe(test_file_name, self.label_to_index, self.task_id, tokenizer)
                

