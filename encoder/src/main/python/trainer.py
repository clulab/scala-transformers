#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import time

from configuration import config
from datasets import Dataset, DatasetDict
from datetime import timedelta
from evaluation_metrics import evaluation_classification_report, evaluate
from task import Task
from token_classifier import TokenClassificationModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer as TransformerTrainer, DataCollatorForTokenClassification

class Trainer():
  def __init__(self):
    # which transformer to use
    self.tokenizer = AutoTokenizer.from_pretrained(config.transformer_name)
    self.config = AutoConfig.from_pretrained(config.transformer_name)

  # main function for training the MTL classifier
  def train(self):
    # the tasks to learn
    ner_task = Task(0, "NER", "data/conll-ner/train.txt", "data/conll-ner/dev.txt", "data/conll-ner/test.txt", self.tokenizer)
    pos_task = Task(1, "POS", "data/pos/train.txt", "data/pos/dev.txt", "data/pos/test.txt", self.tokenizer)
    tasks = [ner_task, pos_task]

    # our own token classifier
    model= TokenClassificationModel.from_pretrained(config.transformer_name, config=config).add_heads(tasks)
    model.summarize_heads()

    # create the formal train/validation/test HF dataset
    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(pd.concat([ner_task.train_df, pos_task.train_df]))
    ds['validation'] = Dataset.from_pandas(pd.concat([ner_task.dev_df, pos_task.dev_df]))
    ds['test'] = Dataset.from_pandas(pd.concat([ner_task.test_df, pos_task.test_df]))

    data_collator = DataCollatorForTokenClassification(self.tokenizer)
    last_checkpoint = None

    # the training loop
    for epoch in range(1, config.epochs + 1):
      print(f'\n\nStarting epoch {epoch}')
      if last_checkpoint != None:
        print(f'Resuming from checkpoint {last_checkpoint}')
              
      training_args = TrainingArguments(
        output_dir=config.model_name,
        log_level='error',
        num_train_epochs=1, # one epoch at a time
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        # evaluation_strategy='epoch',
        do_eval=False, # we will evaluate each task explicitly
        weight_decay=config.weight_decay,
        resume_from_checkpoint = last_checkpoint,
        use_mps_device = config.use_mps_device
      )
      
      trainer = TransformerTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        train_dataset=ds['train'],
        # eval_dataset=ds['validation'],
        tokenizer=self.tokenizer
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

      # keep track of validation scores per epoch
      self.save_stats("epoch_stats.txt", tasks, [ner_acc, pos_acc], epoch)

      # save the transformer encoder + the head for each task
      last_checkpoint = training_args.output_dir + '/mtl_model_epoch' + str(epoch)
      trainer.save_model(last_checkpoint)
      
      # export for JVM
      model.export_model(tasks, self.tokenizer, last_checkpoint + "_export")

  def save_stats(self, fn: str, tasks, accuracies: list[float], epoch: int) -> None:
    with open(fn, 'a') as f:
      f.write(f'{epoch}')
      for i in range(0, len(tasks)):
        f.write(f'\t{tasks[i].task_name}\t{accuracies[i]}')
      f.write('\n')

if __name__ == "__main__":
  Trainer().train()
