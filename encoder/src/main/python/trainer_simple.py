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
from evaluation_metrics import compute_metrics

# main function for training the MTL classifier
def main():
  # which transformer to use
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
  model.summarize_heads()

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

  # Evaluating the intermediate models in this MTL setting is tricky, so we do not do it
  # Instead, the evaluations are handled in averager.py, where tasks are individually evaluated
  training_args = TrainingArguments(
      output_dir=cf.model_name,
      log_level='error',
      num_train_epochs=cf.epochs + 1,
      per_device_train_batch_size=cf.batch_size,
      per_device_eval_batch_size=cf.batch_size,
      save_strategy='epoch',
      #evaluation_strategy='epoch',
      #do_eval=True, 
      weight_decay=cf.weight_decay,
      use_mps_device = cf.use_mps_device
  )
    
  trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      #compute_metrics=compute_metrics,
      train_dataset=ds['train'],
      #eval_dataset=ds['validation'],
      tokenizer=tokenizer
  )
    
  start_time = time.monotonic()
  trainer.train()
  end_time = time.monotonic()
  print(f"Elapsed training time: {timedelta(seconds=end_time - start_time)}")


if __name__ == "__main__":
  main()
