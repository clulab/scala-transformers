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

# create the formal train/validation/test HF dataset
ds = DatasetDict()
ds['train'] = Dataset.from_pandas(pd.concat([
    ner_task.train_df, 
    pos_task.train_df, 
    chunk_task.train_df, 
    deph_task.train_df, 
    depl_task.train_df]))
ds['validation'] = Dataset.from_pandas(pd.concat([
    ner_task.dev_df, 
    pos_task.dev_df, 
    chunk_task.dev_df, 
    deph_task.dev_df,
    depl_task.dev_df]))
ds['test'] = Dataset.from_pandas(pd.concat([
    ner_task.test_df, 
    pos_task.test_df, 
    chunk_task.test_df, 
    deph_task.test_df,
    depl_task.test_df]))

data_collator = OurDataCollator(tokenizer)

# load model from disk
checkpoint = torch.load("bert-base-cased-mtl/checkpoint-500/pytorch_full_model.bin", map_location='cpu')
model.load_state_dict(checkpoint)
model.summarize_heads()
print(f"State dic keys: {model.state_dict().keys()}")
for i in range(5):
    key = f'output_heads.{i}.classifier.weight'
    print(f'{key} = {model.state_dict()[key]}')

training_args = TrainingArguments(
      output_dir=cf.model_name,
      log_level='error',
      num_train_epochs=0, # no training
      do_train = False,
      do_eval = False,
      per_device_train_batch_size=cf.batch_size,
      per_device_eval_batch_size=cf.batch_size,
      weight_decay=cf.weight_decay,
      # resume_from_checkpoint = checkpoint,
      use_mps_device = cf.use_mps_device
)
    
trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      # compute_metrics=compute_metrics,
      train_dataset=ds['train'],
      # eval_dataset=ds['validation'],
      tokenizer=tokenizer
)

# zero training epochs
start_time = time.monotonic()
trainer.train()
end_time = time.monotonic()
print(f"Elapsed time: {timedelta(seconds=end_time - start_time)}")

# evaluate on validation (dev)
model.training_mode = False
ner_acc = evaluate(trainer, ner_task, "NER")
pos_acc = evaluate(trainer, pos_task, "POS")
chunk_acc = evaluate(trainer, chunk_task, "Chunking")
deph_acc = evaluate(trainer, deph_task, "Deps Head")
depl_acc = evaluate(trainer, depl_task, "Deps Label")
macro_acc = (ner_acc + pos_acc + chunk_acc + deph_acc + depl_acc)/5
print(f'Dev macro accuracy for epoch {epoch}: {macro_acc}')