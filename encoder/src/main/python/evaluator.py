#!/usr/bin/env python
# coding: utf-8

import torch

from datasets import Dataset
from names import names
from parameters import parameters
from task import Task
from token_classifier import TokenClassificationModel
from torch import IntTensor, Tensor
from tqdm.notebook import tqdm

class Evaluator:
    def __init__(self, model: TokenClassificationModel) -> None:
        self.model = model
        
    def data_to_tensor(self, dict: dict[str, int]) -> IntTensor: 
        predict_dict = {}
        for key in dict:
            if key in {names.INPUT_IDS, "head_positions"}:
                predict_dict[key] = IntTensor(dict[key]).view(1, len(dict[key]))
                # torch.tensor(IntTensor(dict[key])).view(1, len(dict[key]))
            elif key in {"task_ids"}:
                predict_dict[key] = torch.tensor(dict[key]).view(1)
        return predict_dict

    def labels_to_tensor(self, dict: dict[str, Tensor]) -> Tensor: 
        return torch.tensor(dict["labels"])

    def predict(self, dataset: Dataset) -> tuple[list[float], list[float]]:
        self.model.eval()
        self.model.training_mode = False
        predictions = []
        golds = []
        with torch.no_grad():
            for _, data in tqdm(enumerate(dataset, 0)):
                predict_dict = self.data_to_tensor(data)
                #print("INPUT:")
                #print(predict_dict)
                model_output = self.model(**predict_dict)
                logits = model_output.logits[0]
                #print("PREDICTIONS:")
                #print(logits)
                #print(logits.size())
                pred_labels = torch.argmax(logits, axis = -1)
                #print("PRED LABELS:")
                #print(pred_labels)
                predictions.extend(pred_labels.tolist())
                gold_labels = self.labels_to_tensor(data)
                #print("GOLD LABELS:")
                #print(gold_labels)
                golds.extend(gold_labels.tolist())
        return (golds, predictions)

    # compute accuracy using the model directly
    def evaluation_classification_report(self, task: Task, name: str, useTest: bool = False) -> dict[str, float]:
        print(f"Classification report (useTest = {useTest}) for task {name}:")
        num_labels = task.num_labels
        df = task.dev_df if useTest == False else task.test_df
        ds = Dataset.from_pandas(df)

        golds, predictions = self.predict(ds)
        #print("GOLDS: ", len(golds))
        #print("PREDS: ", len(predictions))

        correct = 0
        total = 0
        for i in range(len(golds)):
            if golds[i] != parameters.ignore_index:
                total = total + 1
                if golds[i] == predictions[i]:
                    correct = correct + 1
        
        acc = correct / total
        print(f"correct = {correct}, total = {total}")
        return {"accuracy": acc}
        
    # compute accuracy on dev or test partition using the given model
    def evaluate_task(self, task: Task) -> float:
        print(f"Evaluating on the validation dataset for task {task.task_name}:")
        acc = self.evaluation_classification_report(task, task.task_name, useTest = False)
        print(acc)
        return acc

    # evaluates self's model and returns macro accuracy on all tasks
    def evaluate(self, tasks: list[Task]) -> float:
        accuracies = [self.evaluate_task(task)["accuracy"] for task in tasks]
        macro_acc = sum(accuracies) / len(accuracies)
        return macro_acc
