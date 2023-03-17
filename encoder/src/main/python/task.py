#!/usr/bin/env python
# coding: utf-8

from data_wrangler import DataWrangler
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class LongTaskDef:
    task_id: int
    task_name: str
    train_file_name: str
    dev_file_name: str
    test_file_name: str
    tokenizer: AutoTokenizer
    dual_mode: bool = False

@dataclass
class ShortTaskDef:
    task_name: str
    local_base_dir: str
    train_file_name: str
    dev_file_name: str
    test_file_name: str
    dual_mode: bool = False

    def to_long_task_def(self, task_id: int, global_base_dir: str, tokenizer: AutoTokenizer) -> LongTaskDef:
        base_dir = f"{global_base_dir}{self.local_base_dir}"
        return LongTaskDef(
            task_id,
            self.task_name,
            f"{base_dir}{self.train_file_name}", 
            f"{base_dir}{self.dev_file_name}",
            f"{base_dir}{self.test_file_name}",
            tokenizer,
            self.dual_mode
        )

class Task:
    def __init__(self, long_task_def: LongTaskDef) -> None:
        self.task_id = long_task_def.task_id
        self.task_name = long_task_def.task_name
        self.dual_mode = long_task_def.dual_mode
        # we need an index of labels first
        self.labels = DataWrangler.read_label_set(long_task_def.train_file_name)
        self.index_to_label = {i:t for i,t in enumerate(self.labels)} 
        self.label_to_index = {t:i for i,t in enumerate(self.labels)} 
        self.num_labels = len(self.index_to_label)
        # create data frames for the datasets
        self.train_df = DataWrangler.read_dataframe(long_task_def.train_file_name, self.label_to_index, self.task_id, long_task_def.tokenizer)
        self.dev_df = DataWrangler.read_dataframe(long_task_def.dev_file_name, self.label_to_index, self.task_id, long_task_def.tokenizer)
        self.test_df = DataWrangler.read_dataframe(long_task_def.test_file_name, self.label_to_index, self.task_id, long_task_def.tokenizer)

        print(f"DF for task {self.task_id}")
        print(self.train_df)
                
    @classmethod
    def mk_tasks(cls, global_base_dir: str, tokenizer: AutoTokenizer, short_task_defs: list[ShortTaskDef]) -> list["Task"]:
        return [
            Task(short_task_def.to_long_task_def(index, global_base_dir, tokenizer)) \
            for index, short_task_def in enumerate(short_task_defs)
        ]
