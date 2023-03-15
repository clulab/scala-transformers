#!/usr/bin/env python
# coding: utf-8

import configuration as cf
import os

from evaluation_metrics import evaluate_model
from task import ShortTaskDef, Task
from token_classifier import TokenClassificationModel
from tokenizer import Tokenizer
from transformers import AutoTokenizer, AutoConfig

class AveragingTrainer():

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.config = AutoConfig.from_pretrained(cf.transformer_name)
        self.tokenizer = tokenizer

    # main function for averaging models coming from different checkpoints
    def train(self, tasks: list[Task]) -> None:
        # create our own token classifier, including the MTL linear layers (or heads)
        model= TokenClassificationModel(self.config, cf.transformer_name).add_heads(tasks)

        all_checkpoints = self.evaluate_checkpoints(model, tasks)
        #all_checkpoints = [("bert-base-cased-mtl/checkpoint-112788", 0.9609160767081839), ("bert-base-cased-mtl/checkpoint-563940", 0.9755393804639914), ("bert-base-cased-mtl/checkpoint-263172", 0.9689283322957432), ("bert-base-cased-mtl/checkpoint-338364", 0.9714441630119305), ("bert-base-cased-mtl/checkpoint-187980", 0.9656722898890078), ("bert-base-cased-mtl/checkpoint-639132", 0.9765498831853234), ("bert-base-cased-mtl/checkpoint-526344", 0.9755484633714315), ("bert-base-cased-mtl/checkpoint-601536", 0.9761621421054925), ("bert-base-cased-mtl/checkpoint-413556", 0.9736679839344381), ("bert-base-cased-mtl/checkpoint-75192", 0.9576241024527834), ("bert-base-cased-mtl/checkpoint-225576", 0.9676488217187262), ("bert-base-cased-mtl/checkpoint-751920", 0.9773871674484382), ("bert-base-cased-mtl/checkpoint-300768", 0.9706321234813376), ("bert-base-cased-mtl/checkpoint-789516", 0.9776340092350553), ("bert-base-cased-mtl/checkpoint-150384", 0.9632114643167207), ("bert-base-cased-mtl/checkpoint-488748", 0.9746556698249005), ("bert-base-cased-mtl/checkpoint-451152", 0.9741960558691349), ("bert-base-cased-mtl/checkpoint-37596", 0.9480660393679876), ("bert-base-cased-mtl/checkpoint-714324", 0.9773336477345376), ("bert-base-cased-mtl/checkpoint-676728", 0.9770617046439647), ("bert-base-cased-mtl/checkpoint-375960", 0.9728387337126444)]
        #all_checkpoints = [("bert-base-cased-mtl/checkpoint-789516", 0.9775049530664692), ("bert-base-cased-mtl/checkpoint-751920", 0.9773593525256891), ("bert-base-cased-mtl/checkpoint-714324", 0.9770666005353906), ("bert-base-cased-mtl/checkpoint-639132", 0.976699004893387), ("bert-base-cased-mtl/checkpoint-676728", 0.9766412444593294)]

        # average the parameters in the top k models
        avg_model = self.average_checkpoints(all_checkpoints, 5, self.config, tasks, tokenizer, "avg", "avg_export")

        # evaluate the averaged model to be sure it works
        macro_acc = self.evaluate_model(avg_model, tasks)
        print(f"Dev macro accuracy for the averaged model: {macro_acc}")

    def evaluate_checkpoints(self, model: TokenClassificationModel, tasks: list[Task]) -> list[tuple[str, float]]:
        best_macro_acc = 0
        best_checkpoint = ""
        all_checkpoints = [] # keeps track of scores for all checkpoints
        directories = (x for x in os.scandir(cf.model_name) if x.is_dir())

        for checkpoint in directories:
            model.from_pretrained(checkpoint.path, ignore_mismatched_sizes=True)
            model.summarize_heads()
            
            # evaluate on validation (dev)
            macro_acc = evaluate_model(model, tasks)
            print(f"Dev macro accuracy for checkpoint {checkpoint}: {macro_acc}")
            
            all_checkpoints.append((checkpoint.path, macro_acc))
            print(f"Current results for all checkpoints: {all_checkpoints}")

            if macro_acc > best_macro_acc:
                best_macro_acc = macro_acc
                best_checkpoint = checkpoint
                print(f"Best checkpoint is {best_checkpoint.path} with a macro accuracy of {best_macro_acc}\n\n")
        
        return all_checkpoints

    def load_model(self, checkpoint: tuple[str, float], config: AutoConfig, tasks: list[Task]) -> TokenClassificationModel:
        model = TokenClassificationModel(config, cf.transformer_name)
        model.add_heads(tasks)
        model.from_pretrained(checkpoint[0], ignore_mismatched_sizes=True)
        return model

    def average_checkpoints(self,
        all_checkpoints: list[tuple[str, float]], k: int, config: AutoConfig, tasks: list[Task],
        tokenizer: AutoTokenizer, dir_to_save: str, dir_to_export: str
    ) -> TokenClassificationModel:
        # sort in descending order of macro accuracy and keep top k
        all_checkpoints.sort(reverse=True, key=lambda x: x[1])
        checkpoints = all_checkpoints[0:k]
        print(f"The top {len(checkpoints)} checkpoints are: {checkpoints}")

        base_dir = cf.model_name
        path_to_save = f"{base_dir}/{dir_to_save}"
        path_to_export = f"{base_dir}/{dir_to_export}"

        print(f"Loading main checkpoint[0] {checkpoints[0]}...")
        main_model = self.load_model(checkpoints[0], config, tasks)
        print("Done loading.")

        self.print_some_params(main_model, "before averaging:")

        for i in range(1, len(checkpoints)):  # Skip 0, which is the main_model.
            print(f"Loading satellite checkpoint[{i}] {checkpoints[i]}...")
            satellite_model = self.load_model(checkpoints[i][0], config, tasks)
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
    tokenizer = Tokenizer.get_pretrained()
    # the tasks to learn
    tasks = Task.mk_tasks("data/", tokenizer, [
        ShortTaskDef("NER",        "conll-ner/", "train.txt",    "dev.txt",    "test.txt"),
        ShortTaskDef("POS",        "pos/",       "train.txt",    "dev.txt",    "test.txt"),
        ShortTaskDef("Chunking",   "chunking/",  "train.txt",    "test.txt",   "test.txt"),
        ShortTaskDef("Deps Head",  "deps-wsj/",  "train.heads",  "dev.heads",  "test.heads"),
        ShortTaskDef("Deps Label", "deps-wsj/",  "train.labels", "dev.labels", "test.labels", dual_mode=True)
    ])
    AveragingTrainer(tokenizer).train(tasks)
