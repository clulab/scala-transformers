#!/usr/bin/env python
# coding: utf-8

import torch 
from torch import nn

from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers import AutoConfig

import os

from configuration import device, transformer_name
from task import Task

# This class is adapted from: https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
class TokenClassificationModel(BertPreTrainedModel):    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.config = config
        self.output_heads = nn.ModuleDict() # these are initialized in add_heads
        self.training_mode = True
        self.init_weights()
        
    def add_heads(self, tasks):
        for task in tasks:
            head = TokenClassificationHead(self.bert.config.hidden_size, task.num_labels, task.task_id, task.dual_mode, self.config.hidden_dropout_prob)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.task_id)] = head
        return self
    
    def summarize_heads(self):
        print(f'Found {len(self.output_heads)} heads')
        for task_id in self.output_heads:
            self.output_heads[task_id].summarize(task_id)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, task_ids=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        sequence_output = outputs[0]

        #print("Keys in kwargs:")
        #for key in kwargs:
        #    print(f'key = {key}')
        
        #print(f'batch size = {len(input_ids)}')
        #print(f'task_ids in this batch: {task_ids}')
        
        # generate specific predictions and losses for each task head
        unique_task_ids_list = torch.unique(task_ids).tolist()
        #print(f'Unique task ids: {unique_task_ids_list}')
        logits = None
        loss_list = []
        for unique_task_id in unique_task_ids_list:
            task_id_filter = task_ids == unique_task_id
            filtered_sequence_output = sequence_output[task_id_filter]
            #filtered_head_positions = None if head_positions is None else head_positions[task_id_filter]
            filtered_labels = None if labels is None else labels[task_id_filter]
            filtered_attention_mask = None if attention_mask is None else attention_mask[task_id_filter]
            #print(f'size of batch for task {unique_task_id} is: {len(filtered_sequence_output)}')
            #print(f'running forward for task {unique_task_id}')
            #print(f'Using task {self.output_heads[str(unique_task_id)].task_id}')

            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                filtered_sequence_output, None,
                # filtered_head_positions, 
                filtered_labels,
                filtered_attention_mask,
            )
            #print(f'done forward for task {unique_task_id}')
            if filtered_labels is not None:
                loss_list.append(task_loss)
                
        loss = None if len(loss_list) == 0 else torch.stack(loss_list)
        #print("batch done")
        #print(f'logits size: {logits.size()}')
                    
        # logits are only used for eval, so we don't save them in training (different task dimensions confuse HF) 
        # at testing time we run one task at a time, so we need to save them then 
        return TokenClassifierOutput(
            loss = loss.mean(),
            logits = None if self.training_mode else logits,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
        )
    

    def save_task(self, task_head, task, task_checkpoint):
        numpy_weights = task_head.classifier.weight.cpu().detach().numpy()
        numpy_bias = task_head.classifier.bias.cpu().detach().numpy()
        labels = task.labels
        #print(f"Shape of weights: {numpy_weights.shape}")
        #print(f"Weights are:\n{numpy_weights}")
        #print(f"Shape of bias: {numpy_bias.shape}")
        #print(f"Bias is: {numpy_bias}")
        #print(f"Labels are: {labels}")

        os.makedirs(task_checkpoint, exist_ok = True)

        self.save_name(task_checkpoint + '/name', task.task_name)
        
        lf = open(task_checkpoint + "/labels", "w")
        for label in labels:
            lf.write(f'{label}\n')
        lf.close()
        
        wf = open(task_checkpoint + "/weights", "w")
        wf.write(f'# {numpy_weights.shape[0]} {numpy_weights.shape[1]}\n')
        for i, x in enumerate(numpy_weights):
            for j, y in enumerate(x):
                wf.write(f'{y} ')
            wf.write('\n')
        wf.close()
        
        bf = open(task_checkpoint + "/biases", "w")
        bf.write(f'# {numpy_bias.shape[0]}\n')
        for i, x in enumerate(numpy_bias):
            bf.write(f'{x} ')
        bf.write('\n')
        bf.close()
    
    def save_name(self, file_name, name):
      f = open(file_name, 'w')
      f.write(f'{name}\n')
      f.close()

    def save_encoder(self, checkpoint, tokenizer):
        orig_words = ["Using", "transformers", "with", "ONNX", "runtime"]
        token_input = tokenizer(orig_words, is_split_into_words = True, return_tensors = "pt")
        # print(token_input)
        token_ids = token_input['input_ids'].to(device)
        
        inputs = (token_ids) 
        input_names = ["token_ids"] 
        output_names = ["sequence_output"]
        
        torch.onnx.export(self.bert,
            inputs,
            checkpoint,
            export_params=True,
            do_constant_folding=True,
            input_names = input_names,
            output_names = output_names,
            opset_version=13, # see: https://chadrick-kwag.net/error-fix-onnxruntime-type-error-type-tensorint64-of-input-parameter-of-operatormin-in-node-is-invalid/
            dynamic_axes = {"token_ids": {1: 'sent length'}}
        )
    
    # exports model in a format friendly for ingestion on the JVM
    def export_model(self, tasks, tokenizer, checkpoint_dir):
        # save the weights/bias in each linear layer
        task_folder = checkpoint_dir + '/tasks'
        os.makedirs(task_folder, exist_ok = True)
        task_counter = 0
        for task in tasks:
            task_checkpoint = task_folder + f'/{task_counter}'
            self.save_task(self.output_heads[str(task.task_id)], task, task_checkpoint)
            task_counter += 1
    
        # save the encoder as an ONNX model
        onnx_checkpoint = checkpoint_dir + '/encoder.onnx'
        self.save_encoder(onnx_checkpoint, tokenizer)
        self.save_name(checkpoint_dir + '/encoder.name', transformer_name)
        

class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, task_id, dual_mode, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.dual_mode = dual_mode
        self.classifier = nn.Linear(
          hidden_size, # if self.dual_mode == False else hidden_size * 2, 
          num_labels
        )
        self.num_labels = num_labels
        self.task_id = task_id
        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
            
    def summarize(self, task_id):
        print(f"Task {self.task_id} with {self.num_labels} labels.")
        print(f'Dropout is {self.dropout}')
        print(f'Classifier layer is {self.classifier}')

    def concatenate(self, sequence_output, head_positions):
      # TODO
      return sequence_output

    def forward(self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs):
        #print(f"sequence_output size = {sequence_output.size()}")
        sequence_output_for_classification = sequence_output # if self.dual_mode == False else concatenate(sequence_output, head_positions)

        sequence_output_dropout = self.dropout(sequence_output_for_classification)
        logits = self.classifier(sequence_output_dropout)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()            
            inputs = logits.view(-1, self.num_labels)
            targets = labels.view(-1)
            loss = loss_fn(inputs, targets)

        return logits, loss


