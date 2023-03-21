#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch

from datasets import Dataset
from names import names
from parameters import parameters
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm

def compute_metrics(eval_pred):
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
            if label_ids[i, j] != parameters.ignore_index:
                y_true.append(label_ids[i][j]) #index_to_label[label_ids[i][j]])
                y_pred.append(pred_ids[i][j]) #index_to_label[pred_ids[i][j]])
    # return computed metrics
    return {"accuracy": accuracy_score(y_true, y_pred)}

def data_to_tensor(dict): 
    predict_dict = {}
    for key in dict:
        if key in {names.INPUT_IDS, "head_positions"}:
            predict_dict[key] = torch.IntTensor(dict[key]).view(1, len(dict[key]))
            # torch.tensor(torch.IntTensor(dict[key])).view(1, len(dict[key]))
        elif key in {"task_ids"}:
            predict_dict[key] = torch.tensor(dict[key]).view(1)
    return predict_dict

def labels_to_tensor(dict): 
    return torch.tensor(dict["labels"])

def predict(model, dataset):
    model.eval()
    model.training_mode = False
    predictions = []
    golds = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataset, 0)):
            predict_dict = data_to_tensor(data)
            #print("INPUT:")
            #print(predict_dict)
            model_output = model(**predict_dict)
            logits = model_output.logits[0]
            #print("PREDICTIONS:")
            #print(logits)
            #print(logits.size())
            pred_labels = torch.argmax(logits, axis = -1)
            #print("PRED LABELS:")
            #print(pred_labels)
            predictions.extend(pred_labels.tolist())
            gold_labels = labels_to_tensor(data)
            #print("GOLD LABELS:")
            #print(gold_labels)
            golds.extend(gold_labels.tolist())
    return (golds, predictions)

# compute accuracy using the model directly
def evaluation_classification_report(model, task, name, useTest=False):
    print(f"Classification report (useTest = {useTest}) for task {name}:")
    num_labels = task.num_labels
    df = task.dev_df if useTest == False else task.test_df
    ds = Dataset.from_pandas(df)

    golds, predictions = predict(model, ds)
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
def evaluate_with_model(model, task):
    print(f"Evaluating on the validation dataset for task {task.task_name}:")
    acc = evaluation_classification_report(model, task, task.task_name, useTest = False)
    print(acc)
    return acc

# evaluates the given model and returns macro accuracy on all tasks
def evaluate_model(model, tasks):
    accuracies = [evaluate_with_model(model, task)["accuracy"] for task in tasks]
    macro_acc = sum(accuracies) / len(accuracies)
    return macro_acc
