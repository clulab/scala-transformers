#!/usr/bin/env python
# coding: utf-8

import os

from basic_trainer import BasicTrainer
from clu_tokenizer import CluTokenizer
from dataclasses import dataclass
from evaluator import Evaluator
from parameters import Parameters
from task import ShortTaskDef, Task
from token_classifier import TokenClassificationModel
from transformers import AutoTokenizer, AutoConfig
from typing import List

@dataclass
class Checkpoint:
    path: str
    accuracy: float

class AveragingTrainer(BasicTrainer):
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        super().__init__(tokenizer)

    # main function for averaging models coming from different checkpoints
    def train(self, tasks: List[Task]) -> None:
        # create our own token classifier, including the MTL linear layers (or heads)
        model = TokenClassificationModel(self.config, Parameters.transformer_name).add_heads(tasks)

        all_checkpoints = self.evaluate_checkpoints(model, tasks)
        #all_checkpoints = [("bert-base-cased-mtl/checkpoint-112788", 0.9609160767081839), ("bert-base-cased-mtl/checkpoint-563940", 0.9755393804639914), ("bert-base-cased-mtl/checkpoint-263172", 0.9689283322957432), ("bert-base-cased-mtl/checkpoint-338364", 0.9714441630119305), ("bert-base-cased-mtl/checkpoint-187980", 0.9656722898890078), ("bert-base-cased-mtl/checkpoint-639132", 0.9765498831853234), ("bert-base-cased-mtl/checkpoint-526344", 0.9755484633714315), ("bert-base-cased-mtl/checkpoint-601536", 0.9761621421054925), ("bert-base-cased-mtl/checkpoint-413556", 0.9736679839344381), ("bert-base-cased-mtl/checkpoint-75192", 0.9576241024527834), ("bert-base-cased-mtl/checkpoint-225576", 0.9676488217187262), ("bert-base-cased-mtl/checkpoint-751920", 0.9773871674484382), ("bert-base-cased-mtl/checkpoint-300768", 0.9706321234813376), ("bert-base-cased-mtl/checkpoint-789516", 0.9776340092350553), ("bert-base-cased-mtl/checkpoint-150384", 0.9632114643167207), ("bert-base-cased-mtl/checkpoint-488748", 0.9746556698249005), ("bert-base-cased-mtl/checkpoint-451152", 0.9741960558691349), ("bert-base-cased-mtl/checkpoint-37596", 0.9480660393679876), ("bert-base-cased-mtl/checkpoint-714324", 0.9773336477345376), ("bert-base-cased-mtl/checkpoint-676728", 0.9770617046439647), ("bert-base-cased-mtl/checkpoint-375960", 0.9728387337126444)]
        #all_checkpoints = [("bert-base-cased-mtl/checkpoint-789516", 0.9775049530664692), ("bert-base-cased-mtl/checkpoint-751920", 0.9773593525256891), ("bert-base-cased-mtl/checkpoint-714324", 0.9770666005353906), ("bert-base-cased-mtl/checkpoint-639132", 0.976699004893387), ("bert-base-cased-mtl/checkpoint-676728", 0.9766412444593294)]

        # average the parameters in the top k models
        avg_model = self.average_checkpoints(all_checkpoints, 5, self.config, tasks, tokenizer, "avg", "avg_export")

        # evaluate the averaged model to be sure it works
        macro_accuracy = Evaluator(avg_model).evaluate(tasks)
        print(f"Dev macro accuracy for the averaged model: {macro_accuracy}")

    def evaluate_checkpoints(self, model: TokenClassificationModel, tasks: List[Task]) -> List[Checkpoint]:
        best_checkpoint = Checkpoint(Parameters.model_name, 0)
        all_checkpoints = [] # keeps track of scores for all checkpoints
        directories = (x for x in os.scandir(Parameters.model_name) if x.is_dir())

        for directory in directories:
            model.from_pretrained(directory.path, ignore_mismatched_sizes=True)
            model.summarize_heads()
            
            # evaluate on validation (dev)
            macro_accuracy = Evaluator(model).evaluate(tasks)
            print(f"Dev macro accuracy for checkpoint {directory}: {macro_accuracy}")
            
            all_checkpoints.append(Checkpoint(directory.path, macro_accuracy))
            print(f"Current results for all checkpoints: {all_checkpoints}")

            if macro_accuracy > best_checkpoint.accuracy:
                best_checkpoint = Checkpoint(directory.path, macro_accuracy)
                print(f"Best checkpoint is {best_checkpoint.path} with a macro accuracy of {best_checkpoint.accuracy}\n\n")
        
        return all_checkpoints

    def load_model(self, checkpoint: Checkpoint, config: AutoConfig, tasks: List[Task]) -> TokenClassificationModel:
        model = TokenClassificationModel(config, Parameters.transformer_name)
        model.add_heads(tasks)
        model.from_pretrained(checkpoint.path, ignore_mismatched_sizes=True)
        return model

    def average_checkpoints(self,
        all_checkpoints: List[Checkpoint], k: int, config: AutoConfig, tasks: List[Task],
        tokenizer: AutoTokenizer, dir_to_save: str, dir_to_export: str
    ) -> TokenClassificationModel:
        # sort in descending order of macro accuracy and keep top k
        all_checkpoints.sort(reverse=True, key=lambda checkpoint: checkpoint.accuracy)
        checkpoints = all_checkpoints[0:k]
        print(f"The top {len(checkpoints)} checkpoints are: {checkpoints}")

        base_dir = Parameters.model_name
        path_to_save = f"{base_dir}/{dir_to_save}"
        path_to_export = f"{base_dir}/{dir_to_export}"

        print(f"Loading main checkpoint[0] {checkpoints[0]}...")
        main_model = self.load_model(checkpoints[0], config, tasks)
        print("Done loading.")

        self.print_some_params(main_model, "before averaging:")

        for i in range(1, len(checkpoints)):  # Skip 0, which is the main_model.
            print(f"Loading satellite checkpoint[{i}] {checkpoints[i]}...")
            satellite_model = self.load_model(checkpoints[i], config, tasks)
            self.print_some_params(satellite_model, "satellite model:")

            print("Adding its parameter weights to the main model...")
            for key, value in main_model.state_dict().items():
                if value.data.type() != "torch.LongTensor":
                    value.data += satellite_model.state_dict()[key].data.clone()
            print("Done adding")

        self.print_some_params(main_model, "after summing:")
        if len(checkpoints) > 1:
            print("Computing average weights...")
            for value in main_model.state_dict().values():
                if value.data.type() != "torch.LongTensor":
                    value.data /= len(checkpoints)
            print("Done computing.")
        
        self.print_some_params(main_model, "after averaging:")
        print("Saving averaged model...")
        main_model.save_pretrained(path_to_save)
        main_model.export_model(tasks, tokenizer, path_to_export)
        print("Done saving.")
        return main_model

    def print_some_params(self, model: TokenClassificationModel, msg: str) -> None:
        print(msg)
        for i in range(4, 5):
            key = f"output_heads.{i}.classifier.weight"
            print(f"{key} = {model.state_dict()[key]}")

if __name__ == "__main__":
    tokenizer = CluTokenizer.get_pretrained()
    # the tasks to learn
    tasks = Task.mk_tasks("data/", tokenizer, [
        ShortTaskDef("NER",        "conll-ner/", "train.txt",    "dev.txt",    "test.txt"),
        ShortTaskDef("POS",        "pos/",       "train.txt",    "dev.txt",    "test.txt"),
        ShortTaskDef("Chunking",   "chunking/",  "train.txt",    "test.txt",   "test.txt"),
        #ShortTaskDef("Deps Head",  "deps-wsj/",  "train.heads",  "dev.heads",  "test.heads"),
        #ShortTaskDef("Deps Label", "deps-wsj/",  "train.labels", "dev.labels", "test.labels", dual_mode=True)
        ShortTaskDef("Deps Head",  "deps-combined/", "wsjtrain-wsjdev-geniatrain-geniadev.heads",  "test.heads",  "test.heads"),
        ShortTaskDef("Deps Label", "deps-combined/", "wsjtrain-wsjdev-geniatrain-geniadev.labels", "test.labels", "test.labels", dual_mode=True)
    ])
    AveragingTrainer(tokenizer).train(tasks)
