#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

import time
from datetime import timedelta

import os

import configuration as cf
from task import Task
from token_classifier import TokenClassificationModel
from dual_data_collator import OurDataCollator
from evaluation_metrics import evaluation_classification_report, evaluate, evaluate_with_model

# main function for averaging models coming from different checkpoints
print(f'Loading tokenizer named "{cf.transformer_name}"...')
tokenizer = AutoTokenizer.from_pretrained(cf.transformer_name, model_input_names=["input_ids", "token_type_ids", "attention_mask"])
config = AutoConfig.from_pretrained(cf.transformer_name)

# the tasks to learn
ner_task = Task(0, "NER", "data/conll-ner/train.txt", "data/conll-ner/dev.txt", "data/conll-ner/test.txt", tokenizer)
pos_task = Task(1, "POS", "data/pos/train.txt", "data/pos/dev.txt", "data/pos/test.txt", tokenizer)
chunk_task = Task(2, "Chunking", "data/chunking/train.txt", "data/chunking/test.txt", "data/chunking/test.txt", tokenizer)
deph_task = Task(3, "Deps Head", "data/deps-wsj/train.heads", "data/deps-wsj/dev.heads", "data/deps-wsj/test.heads", tokenizer)
depl_task = Task(4, "Deps Label", "data/deps-wsj/train.labels", "data/deps-wsj/dev.labels", "data/deps-wsj/test.labels", tokenizer, dual_mode = True)
tasks = [ner_task, pos_task, chunk_task, deph_task, depl_task]

# create our own token classifier, including the MTL linear layers (or heads)
model= TokenClassificationModel(config)
model.add_heads(tasks)

best_macro_acc = 0
best_checkpoint = ""
all_checkpoints = [] # keeps track of scores for all checkpoints

base_dir = "bert-base-cased-mtl/"
for it in os.scandir(base_dir):
    if it.is_dir():
      checkpoint = it
      model.from_pretrained(checkpoint.path, ignore_mismatched_sizes=True)
      model.summarize_heads()

      # evaluate on validation (dev)
      ner_acc = evaluate_with_model(model, ner_task, "NER")
      pos_acc = evaluate_with_model(model, pos_task, "POS")
      chunk_acc = evaluate_with_model(model, chunk_task, "Chunking")
      deph_acc = evaluate_with_model(model, deph_task, "Deps Head")
      depl_acc = evaluate_with_model(model, depl_task, "Deps Label")
      macro_acc = (ner_acc['accuracy'] + pos_acc['accuracy'] + chunk_acc['accuracy'] + deph_acc['accuracy'] + depl_acc['accuracy'])/5
      print(f'Dev macro accuracy for checkpoint {checkpoint}: {macro_acc}')

      all_checkpoints.append((checkpoint.path, macro_acc))
      print(f"Current results for all checkpoints: {all_checkpoints}")

      if macro_acc > best_macro_acc:
         best_macro_acc = macro_acc
         best_checkpoint = checkpoint
         print(f"Best checkpoint is {best_checkpoint.path} with a macro accuracy of {best_macro_acc}\n\n")

# sort in descending order of macro accuracy and keep top k
all_checkpoints.sort(reverse=True, key=sort_func)
checkpoints_to_average = all_checkpoints[0:5]

# average the parameters in the top k models
# TODO

def sort_func(cp):
   return cp[1]
