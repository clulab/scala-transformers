#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from datasets import Dataset

from configuration import ignore_index

def compute_metrics(eval_pred):
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

# compute accuracy
def evaluation_classification_report(trainer, task, name, useTest=False):
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
        target_names=target_names
    )
    print(report)
    print(f'Locally computed accuracy: {accuracy}')
    return accuracy

# compute loss and accuracy
def evaluate(trainer, task, name):
    print(f"Evaluating on the validation dataset for task {name}:")
    #ds = Dataset.from_pandas(task.dev_df)
    #scores = trainer.evaluate(ds)
    acc = evaluation_classification_report(trainer, task, name, useTest = False)
    return acc

