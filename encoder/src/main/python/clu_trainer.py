#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from basic_trainer import BasicTrainer
from clu_timer import CluTimer
from clu_tokenizer import CluTokenizer
from datasets import Dataset
from dual_data_collator import DualDataCollator
from names import Names
from parameters import Parameters
from task import ShortTaskDef, Task
from sklearn.metrics import accuracy_score
from token_classifier import TokenClassificationModel
from transformers import AutoTokenizer, EvalPrediction, TrainingArguments, Trainer
from typing import Dict, List

class CluTrainer(BasicTrainer):
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        super().__init__(tokenizer)

    # main function for training the MTL classifier
    def train(self, tasks: List[Task]) -> None:
        # our own token classifier
        model = TokenClassificationModel(self.config, Parameters.transformer_name).add_heads(tasks)
        model.summarize_heads()

        # create the formal train/validation/test HF dataset
        train_ds = Dataset.from_pandas(pd.concat([task.train_df for task in tasks]))
        #validation_ds = Dataset.from_pandas(pd.concat([task.dev_df for task in tasks]))
        #test_ds = Dataset.from_pandas(pd.concat([task.test_df for task in tasks]))

        data_collator = DualDataCollator(self.tokenizer)

        # Evaluating the intermediate models in this MTL setting is tricky, so we do not do it
        # Instead, the evaluations are handled in averager.py, where tasks are individually evaluated
        training_args = TrainingArguments(
            output_dir=Parameters.model_name,
            log_level="error",
            num_train_epochs=Parameters.epochs + 1,
            per_device_train_batch_size=Parameters.batch_size,
            per_device_eval_batch_size=Parameters.batch_size,
            save_strategy="epoch",
            #evaluation_strategy="epoch",
            #do_eval=True, 
            weight_decay=Parameters.weight_decay,
            use_mps_device = Parameters.use_mps_device,
            no_cuda = not Parameters.use_cuda_device
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            # compute_metrics=lambda eval_pred: self.compute_metrics(eval_pred),
            train_dataset=train_ds,
            #eval_dataset=validation_ds,
            tokenizer=self.tokenizer
        )
        
        CluTimer.time(
            lambda: trainer.train()
        )

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        #print("GOLDS: ", eval_pred.label_ids)
        #print("PREDS: ", eval_pred.predictions)
        # gold labels
        label_ids = eval_pred.label_ids
        # predictions
        pred_ids = np.argmax(eval_pred.predictions, axis=-1)
        # collect gold and predicted labels, ignoring ignore_index label
        y_true, y_pred = [], []
        batch_size, seq_len = pred_ids.shape
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != Parameters.ignore_index:
                    y_true.append(label_ids[i][j]) #index_to_label[label_ids[i][j]])
                    y_pred.append(pred_ids[i][j]) #index_to_label[pred_ids[i][j]])
        # return computed metrics
        return {Names.ACCURACY: accuracy_score(y_true, y_pred)}


if __name__ == "__main__":
    tokenizer = CluTokenizer.get_pretrained()
    # the tasks to learn
    tasks = Task.mk_tasks("data/", tokenizer, [
        ShortTaskDef("NER",       "conll-ner/", "train.txt",    "dev.txt",    "test.txt"),
        ShortTaskDef("POS",             "pos/", "train.txt",    "dev.txt",    "test.txt"),
        ShortTaskDef("Chunking",   "chunking/", "train.txt",    "test.txt",   "test.txt"), # this dataset has no dev
        ShortTaskDef("Deps Head",  "deps-combined/", "wsjtrain-wsjdev-geniatrain-geniadev.heads",  "dev.heads",  "test.heads"),
        ShortTaskDef("Deps Label", "deps-combined/", "wsjtrain-wsjdev-geniatrain-geniadev.labels", "dev.labels", "test.labels", dual_mode=True)
        #ShortTaskDef("Deps Head",  "deps-wsj/", "train.heads",  "dev.heads",  "test.heads"),
        #ShortTaskDef("Deps Label", "deps-wsj/", "train.labels", "dev.labels", "test.labels", dual_mode=True)
    ])
    CluTrainer(tokenizer).train(tasks)
