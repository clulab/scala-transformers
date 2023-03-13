#!/usr/bin/env python
# coding: utf-8

import configuration as cf
import pandas as pd
import time

from datasets import Dataset
from datetime import timedelta
from dual_data_collator import OurDataCollator
from evaluation_metrics import compute_metrics
from task import ShortTaskDef, Task
from token_classifier import TokenClassificationModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer

# main function for training the MTL classifier
def main(small: bool = False) -> None:
    # which transformer to use
    print(f'Loading tokenizer named "{cf.transformer_name}"...')
    tokenizer = AutoTokenizer.from_pretrained(cf.transformer_name, model_input_names=["input_ids", "token_type_ids", "attention_mask"])
    config = AutoConfig.from_pretrained(cf.transformer_name)

    # the tasks to learn
    tasks = Task.mk_tasks("data/", tokenizer, [
        ShortTaskDef("NER",       "conll-ner/", "train_small.txt",    "dev.txt",          "test.txt"),
        ShortTaskDef("POS",             "pos/", "train_small.txt",    "dev.txt",          "test.txt"),
        ShortTaskDef("Chunking",   "chunking/", "train_small.txt",    "test_small.txt",   "test_small.txt"),
        ShortTaskDef("Deps Head",  "deps-wsj/", "train_small.heads",  "dev_small.heads",  "test_small.heads"),
        ShortTaskDef("Deps Label", "deps-wsj/", "train_small.labels", "dev_small.labels", "test_small.labels", dual_mode = True)
    ])

    # our own token classifier
    model = TokenClassificationModel(config, cf.transformer_name).add_heads(tasks)
    model.summarize_heads()

    # create the formal train/validation/test HF dataset
    train_ds = Dataset.from_pandas(pd.concat([task.train_df for task in tasks]))
    validation_ds = Dataset.from_pandas(pd.concat([task.dev_df for task in tasks]))
    test_ds = Dataset.from_pandas(pd.concat([task.test_df for task in tasks]))

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
        train_dataset=train_ds,
        #eval_dataset=validation_ds,
        tokenizer=tokenizer
    )
    
    start_time = time.monotonic()
    trainer.train()
    end_time = time.monotonic()
    print(f"Elapsed training time: {timedelta(seconds=end_time - start_time)}")


if __name__ == "__main__":
    main()
