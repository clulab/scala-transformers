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

import configuration as cf
from task import Task
from token_classifier import TokenClassificationModel
from dual_data_collator import OurDataCollator
from evaluation_metrics import evaluation_classification_report, evaluate

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

# our own token classifier
model= TokenClassificationModel.from_pretrained(cf.transformer_name, config=config, ignore_mismatched_sizes=True).add_heads(tasks)

# load model from disk
checkpoint = torch.load("bert-base-cased-mtl/checkpoint-413556/pytorch_full_model.bin", map_location='cpu')
model.load_state_dict(checkpoint)
model.summarize_heads()

# evaluate on validation (dev)
ner_acc = evaluate(model, ner_task, "NER")
pos_acc = evaluate(model, pos_task, "POS")
chunk_acc = evaluate(model, chunk_task, "Chunking")
deph_acc = evaluate(model, deph_task, "Deps Head")
depl_acc = evaluate(model, depl_task, "Deps Label")
macro_acc = (ner_acc + pos_acc + chunk_acc + deph_acc + depl_acc)/5
print(f'Dev macro accuracy for epoch {epoch}: {macro_acc}')

