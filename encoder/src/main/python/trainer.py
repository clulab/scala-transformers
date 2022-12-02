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
  print(f'Loading tokenizer named "{cf.transformer_name}"...')
  tokenizer = AutoTokenizer.from_pretrained(cf.transformer_name, model_input_names=["input_ids", "token_type_ids", "attention_mask", "head_positionszzz"])
  # tokenizer.model_input_names = ["input_ids", "token_type_ids", "attention_mask", "head_positionszzz"]
  config = AutoConfig.from_pretrained(cf.transformer_name)

  # the tasks to learn
  #ner_task = Task(0, "NER", "data/conll-ner/train.txt", "data/conll-ner/dev.txt", "data/conll-ner/test.txt", tokenizer)
  #pos_task = Task(1, "POS", "data/pos/train.txt", "data/pos/dev.txt", "data/pos/test.txt", tokenizer)
  #chunk_task = Task(2, "Chunking", "data/chunking/train.txt", "data/chunking/test.txt", "data/chunking/test.txt", tokenizer)
  #deph_task = Task(3, "Deps Head", "data/deps-wsj/train.heads", "data/deps-wsj/dev.heads", "data/deps-wsj/test.heads", tokenizer)
  depl_task = Task(0, "Deps Label", "data/deps-wsj/train.labels", "data/deps-wsj/dev.labels", "data/deps-wsj/test.labels", tokenizer, dual_mode = True)
  tasks = [depl_task] #ner_task, pos_task, chunk_task, deph_task, depl_task]

  # our own token classifier
  model= TokenClassificationModel.from_pretrained(cf.transformer_name, config=config, ignore_mismatched_sizes=True).add_heads(tasks)
  model.summarize_heads()

  # create the formal train/validation/test HF dataset
  ds = DatasetDict()
  ds['train'] = Dataset.from_pandas(pd.concat([
    #ner_task.train_df, 
    #pos_task.train_df, 
    #chunk_task.train_df, 
    #deph_task.train_df, 
    depl_task.train_df]))
  ds['validation'] = Dataset.from_pandas(pd.concat([
    #ner_task.dev_df, 
    #pos_task.dev_df, 
    #chunk_task.dev_df, 
    #deph_task.dev_df,
    depl_task.dev_df]))
  ds['test'] = Dataset.from_pandas(pd.concat([
    #ner_task.test_df, 
    #pos_task.test_df, 
    #chunk_task.test_df, 
    #deph_task.test_df,
    depl_task.test_df]))

  data_collator = DataCollatorForTokenClassification(tokenizer)
  # print(ds['train'][0])
  # print(data_collator(ds['train'][:1]))
  # exit()
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
    model.training_mode = False
    #ner_acc = evaluate(trainer, ner_task, "NER")
    #pos_acc = evaluate(trainer, pos_task, "POS")
    #chunk_acc = evaluate(trainer, chunk_task, "Chunking")
    #deph_acc = evaluate(trainer, deph_task, "Deps Head")
    depl_acc = evaluate(trainer, depl_task, "Deps Label")
    macro_acc = depl_acc # (ner_acc + pos_acc + chunk_acc + deph_acc + depl_acc)/5
    print(f'Dev macro accuracy for epoch {epoch}: {macro_acc}')
    model.training_mode = True

    # keep track of validation scores per epoch
    save_stats("epoch_stats.txt", tasks,  
      [depl_acc], # [ner_acc, pos_acc, chunk_acc, deph_acc, depl_acc], 
      macro_acc, epoch
    )

    # save the transformer encoder + the head for each task
    last_checkpoint = training_args.output_dir + '/mtl_model_epoch' + str(epoch)
    trainer.save_model(last_checkpoint)
    
    # export for JVM
    model.export_model(tasks, tokenizer, last_checkpoint + "_export")

def save_stats(fn, tasks, accuracies, macro_acc, epoch):
  f = open(fn, 'a')
  f.write(f'{epoch}')
  for i in range(0, len(tasks)):
    f.write(f'\t{tasks[i].task_name}\t{accuracies[i]}')
  f.write(f'\tmacro\t{macro_acc}')
  f.write('\n')
  f.close()

if __name__ == "__main__":
  main()
