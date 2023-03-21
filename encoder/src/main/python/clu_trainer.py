#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from basic_trainer import BasicTrainer
from clu_tokenizer import CluTokenizer
from datasets import Dataset
from dual_data_collator import DualDataCollator
from evaluator import compute_metrics
from parameters import parameters
from task import ShortTaskDef, Task
from clu_timer import CluTimer
from token_classifier import TokenClassificationModel
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer

class CluTrainer(BasicTrainer):
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        super().__init__(tokenizer)

    # main function for training the MTL classifier
    def train(self, tasks: list[Task]) -> None:
        # our own token classifier
        model = TokenClassificationModel(self.config, parameters.transformer_name).add_heads(tasks)
        model.summarize_heads()

        # create the formal train/validation/test HF dataset
        train_ds = Dataset.from_pandas(pd.concat([task.train_df for task in tasks]))
        #validation_ds = Dataset.from_pandas(pd.concat([task.dev_df for task in tasks]))
        #test_ds = Dataset.from_pandas(pd.concat([task.test_df for task in tasks]))

        data_collator = DualDataCollator(self.tokenizer)

        # Evaluating the intermediate models in this MTL setting is tricky, so we do not do it
        # Instead, the evaluations are handled in averager.py, where tasks are individually evaluated
        training_args = TrainingArguments(
            output_dir=parameters.model_name,
            log_level="error",
            num_train_epochs=parameters.epochs + 1,
            per_device_train_batch_size=parameters.batch_size,
            per_device_eval_batch_size=parameters.batch_size,
            save_strategy="epoch",
            #evaluation_strategy="epoch",
            #do_eval=True, 
            weight_decay=parameters.weight_decay,
            use_mps_device = parameters.use_mps_device
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            #compute_metrics=compute_metrics,
            train_dataset=train_ds,
            #eval_dataset=validation_ds,
            tokenizer=self.tokenizer
        )
        
        CluTimer.time(
            lambda: trainer.train()
        )


if __name__ == "__main__":
    tokenizer = CluTokenizer.get_pretrained()
    # the tasks to learn
    tasks = Task.mk_tasks("data/", tokenizer, [
        ShortTaskDef("NER",       "conll-ner/", "train.txt",    "dev.txt",    "test.txt"),
        ShortTaskDef("POS",             "pos/", "train.txt",    "dev.txt",    "test.txt"),
        ShortTaskDef("Chunking",   "chunking/", "train.txt",    "test.txt",   "test.txt"),
        ShortTaskDef("Deps Head",  "deps-wsj/", "train.heads",  "dev.heads",  "test.heads"),
        ShortTaskDef("Deps Label", "deps-wsj/", "train.labels", "dev.labels", "test.labels", dual_mode=True)
    ])
    CluTrainer(tokenizer).train(tasks)
