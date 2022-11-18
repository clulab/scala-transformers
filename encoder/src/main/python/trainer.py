#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict

import time
from datetime import timedelta

import configuration as cf
from task import Task
from token_classifier import TokenClassificationModel
from evaluation_metrics import evaluation_classification_report, evaluate

# main function for training the MTL classifier
def main():
  # which transformer to use
  tokenizer = AutoTokenizer.from_pretrained(cf.transformer_name)
  config = AutoConfig.from_pretrained(cf.transformer_name)

  # the tasks to learn
  ner_task = Task(0, "NER", "data/conll-ner/train.txt", "data/conll-ner/dev.txt", "data/conll-ner/test.txt", tokenizer)
  pos_task = Task(1, "POS", "data/pos/train.txt", "data/pos/dev.txt", "data/pos/test.txt", tokenizer)
  tasks = [ner_task, pos_task]

  # our own token classifier
  model= TokenClassificationModel.from_pretrained(cf.transformer_name, config=config).add_heads(tasks)
  model.summarize_heads()

  # create the formal train/validation/test HF dataset
  ds = DatasetDict()
  ds['train'] = Dataset.from_pandas(pd.concat([ner_task.train_df, pos_task.train_df]))
  ds['validation'] = Dataset.from_pandas(pd.concat([ner_task.dev_df, pos_task.dev_df]))
  ds['test'] = Dataset.from_pandas(pd.concat([ner_task.test_df, pos_task.test_df]))

  data_collator = DataCollatorForTokenClassification(tokenizer)
  last_checkpoint = None

  # the training loop
  for epoch in range(1, cf.epochs + 1):
    print(f'\n\nStarting epoch {epoch}')
    if last_checkpoint != None:
      print(f'Resuming from checkpoint {last_checkpoint}')
            
    training_args = TrainingArguments(
      output_dir=cf.model_name,
      log_level='error',
      num_train_epochs=1, # one epoch at a time
      per_device_train_batch_size=cf.batch_size,
      per_device_eval_batch_size=cf.batch_size,
      # evaluation_strategy='epoch',
      do_eval=False, # we will evaluate each task explicitly
      weight_decay=cf.weight_decay,
      resume_from_checkpoint = last_checkpoint,
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
    
    # one training epoch
    start_time = time.monotonic()
    trainer.train()
    end_time = time.monotonic()
    print(f"Elapsed time for epoch {epoch}: {timedelta(seconds=end_time - start_time)}")

    # evaluate on validation (dev)
    ner_acc = evaluate(trainer, ner_task, "NER")
    pos_acc = evaluate(trainer, pos_task, "POS")
    macro_acc = (ner_acc + pos_acc)/2
    print(f'Dev macro accuracy for epoch {epoch}: {macro_acc}')

    # save the transformer encoder + the head for each task
    last_checkpoint = training_args.output_dir + '/mtl_model_epoch' + str(epoch)
    trainer.save_model(last_checkpoint)
    
    # export for JVM
    model.export_model(tasks, tokenizer, last_checkpoint + "_export")

  # evaluate on test using the latest model
  print("Evaluating on test...")
  ner_acc = evaluation_classification_report(trainer, ner_task, "NER", useTest = True)
  pos_acc = evaluation_classification_report(trainer, pos_task, "POS", useTest = True)
  macro_acc = (ner_acc + pos_acc)/2
  print(f"MTL macro accuracy on test: {macro_acc}")

if __name__ == "__main__":
  main()
