#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from datasets import Dataset

from configuration import ignore_index, device

from tqdm.notebook import tqdm

def compute_metrics(eval_pred):
    print("GOLDS: ", eval_pred.label_ids)
    print("PREDS: ", eval_pred.predictions)
    # gold labels
    label_ids = eval_pred.label_ids
    # predictions
    pred_ids = np.argmax(eval_pred.predictions, axis=-1)
    # collect gold and predicted labels, ignoring ignore_index label
    y_true, y_pred = [], []
    batch_size, seq_len = pred_ids.shape
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != ignore_index:
                y_true.append(label_ids[i][j]) #index_to_label[label_ids[i][j]])
                y_pred.append(pred_ids[i][j]) #index_to_label[pred_ids[i][j]])
    # return computed metrics
    return {'accuracy': accuracy_score(y_true, y_pred)}

def data_to_tensor(dict): 
    predict_dict = {}
    for key in dict:
        if key in {'input_ids', 'head_positions'}:
            predict_dict[key] = torch.IntTensor(dict[key]).view(1, len(dict[key]))
            # torch.tensor(torch.IntTensor(dict[key])).view(1, len(dict[key]))
        elif key in {'task_ids'}:
            predict_dict[key] = torch.tensor(dict[key]).view(1)
    return predict_dict

def labels_to_tensor(dict): 
    return torch.tensor(dict['labels'])

def predict(model, dataset):
    model.eval()
    model.training_mode = False
    predictions = []
    golds = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataset, 0)):
            predict_dict = data_to_tensor(data)
            #print('INPUT:')
            #print(predict_dict)
            model_output = model(**predict_dict)
            logits = model_output.logits[0]
            #print('PREDICTIONS:')
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
    print("GOLDS: ", len(golds))
    print("PREDS: ", len(predictions))

    correct = 0
    total = 0
    for i in range(len(golds)):
        if golds[i] != ignore_index:
            total = total + 1
            if golds[i] == predictions[i]:
                correct = correct + 1
    
    acc = correct / total
    print(f"correct = {correct}, total = {total}")
    return {'accuracy': acc}
    
# compute accuracy using a trainer
def evaluation_classification_report_with_trainer(trainer, task, name, useTest=False):
    print(f"Classification report (useTest = {useTest}) for task {name}:")
    num_labels = task.num_labels
    df = task.dev_df if useTest == False else task.test_df
    ds = Dataset.from_pandas(df)
    output = trainer.predict(ds)
    label_ids = output.label_ids.reshape(-1)
    predictions = output.predictions.reshape(-1, num_labels)
    predictions = np.argmax(predictions, axis=-1)
    mask = label_ids != ignore_index

    y_true = label_ids[mask]
    y_pred = predictions[mask]
    target_names = [task.index_to_label.get(ele, "") for ele in range(num_labels)]
    print(target_names)
    
    total = 0
    correct = 0
    for(t, p) in zip(y_true, y_pred):
        total = total + 1
        if t == p:
            correct = correct + 1
    accuracy = correct / total
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        labels = range(0, len(task.index_to_label))
    )
    print(report)
    print(f'Locally computed accuracy: {accuracy}')
    return accuracy    

# compute loss and accuracy
def evaluate(trainer, task, name):
    print(f"Evaluating on the validation dataset for task {name}:")
    # uncomment these two lines if you want to compute loss on dev
    #ds = Dataset.from_pandas(task.dev_df)
    #scores = trainer.evaluate(ds)
    acc = evaluation_classification_report_with_trainer(trainer, task, name, useTest = False)
    print(acc)
    return acc

def evaluate_with_model(model, task):
    print(f"Evaluating on the validation dataset for task {task.task_name}:")
    # uncomment these two lines if you want to compute loss on dev
    #ds = Dataset.from_pandas(task.dev_df)
    #scores = trainer.evaluate(ds)
    acc = evaluation_classification_report(model, task, task.task_name, useTest = False)
    print(acc)
    return acc

