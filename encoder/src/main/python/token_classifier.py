#!/usr/bin/env python
# coding: utf-8

import os
import torch

from file_utils import FileUtils
from names import names
from tensor_filter import TensorFilter
from parameters import parameters
from task import Task
from torch import nn, Tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Callable, Optional, Union

# This class is adapted from: https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
class TokenClassificationModel(PreTrainedModel):    
    def __init__(self, config: AutoConfig, transformer_name: str) -> None:
        super().__init__(config)
        self.encoder: AutoModel = AutoModel.from_pretrained(transformer_name, config = config) 
        self.config: AutoConfig = config
        self.output_heads: nn.ModuleDict = nn.ModuleDict() # these are initialized in add_heads
        self.training_mode: bool = True

    def add_heads(self, tasks: list[Task]) -> "TokenClassificationModel":
        for task in tasks:
            head = TokenClassificationHead(self.encoder.config.hidden_size, task.num_labels, task.task_id, task.dual_mode, self.config.hidden_dropout_prob)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.task_id)] = head
        # initialize the weights in all heads
        self._init_weights()
        return self
    
    def summarize_heads(self) -> None:
        print(f"Found {len(self.output_heads)} heads")
        for task_id, head in self.output_heads.items():
            head.summarize(task_id)

    def _init_weights(self) -> None:
        for head in self.output_heads.values():
            head._init_weights()
    
    def forward(self,
        input_ids: Tensor = None, attention_mask: Tensor = None, token_type_ids: Tensor = None,
        labels: Tensor = None, head_positions: Tensor = None, task_ids: Tensor = None, **kwargs: str
    ) -> TokenClassifierOutput:
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        sequence_output = outputs[0]

        #print("Keys in kwargs:")
        #for key in kwargs:
        #    print(f"key = {key}")
        
        #print(f"batch size = {len(input_ids)}")
        #print(f"task_ids in this batch: {task_ids}")
        
        # generate specific predictions and losses for each task head
        unique_task_ids = torch.unique(task_ids).tolist()
        #print(f"Unique task ids: {unique_task_ids_list}")
        logits = None
        loss_list = []
        for unique_task_id in unique_task_ids:
            task_id_filter = TensorFilter(task_ids == unique_task_id)
            #print(f"sequence_output = {sequence_output}")
            #print(f"task_id_filter = {task_id_filter}")
            filtered_sequence_output = task_id_filter.filter(sequence_output)
            filtered_head_positions = task_id_filter.filter(head_positions)
            filtered_labels = task_id_filter.filter(labels)
            filtered_attention_mask = task_id_filter.filter(attention_mask)
            #print(f"size of batch for task {unique_task_id} is: {len(filtered_sequence_output)}")
            #print(f"running forward for task {unique_task_id}")
            #print(f"Using task {self.output_heads[str(unique_task_id)].task_id}")

            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                filtered_sequence_output, None,
                filtered_head_positions, 
                filtered_labels,
                filtered_attention_mask
            )
            #print(f"done forward for task {unique_task_id}")
            if filtered_labels is not None:
                loss_list.append(task_loss) #lost_list is empty, so loss becomes empty
                
        loss = None if len(loss_list) == 0 else torch.stack(loss_list)
        #print("batch done")
        #print(f"logits size: {logits.size()}")
                    
        # logits are only used for eval, so we don't save them in training (different task dimensions confuse HF) 
        # at testing time we run one task at a time, so we need to save them then 
        return TokenClassifierOutput(
            loss=None if labels is None else loss.mean(),
            logits=None if self.training_mode else logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def save_pretrained(self,
        save_directory: str, is_main_process: bool = True, state_dict:  Optional[dict] = None, save_function: Callable = torch.save,
        push_to_hub: bool = False, max_shard_size: str = "10GB", safe_serialization: bool = False, **kwargs: str
    ) -> None:
        print(f"Saving model to folder {save_directory}")
        print("super.save_pretrained started...")
        super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, **kwargs)
        print("super.save_pretrained done.")
        print("Saving pickle of complete model...")
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
        print("pickle saving done.")
        #super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, **kwargs)
        #for i in range(5):
        #  key = f"output_heads.{i}.classifier.weight"
        #  print(f"{key} = {self.state_dict()[key]}")

    def from_pretrained(self, pretrained_model_name_or_path: str, *model_args, **kwargs) -> None:
        # the line below is not needed. We initialize the weights from the pickle; the rest are default values
        # super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # HF does not initialize our MTL linear layers, so we have to do it explicitly
        if os.path.isdir(pretrained_model_name_or_path):
            print("Loading full model from pickle...")
            checkpoint = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin", map_location="cpu")
            self.load_state_dict(checkpoint)    
            print("Done loading.")

    def export_task(self, task_head, task: Task, task_checkpoint) -> None:
        numpy_weights = task_head.classifier.weight.cpu().detach().numpy()
        numpy_bias = task_head.classifier.bias.cpu().detach().numpy()
        labels = task.labels
        #print(f"Shape of weights: {numpy_weights.shape}")
        #print(f"Weights are:\n{numpy_weights}")
        #print(f"Shape of bias: {numpy_bias.shape}")
        #print(f"Bias is: {numpy_bias}")
        #print(f"Labels are: {labels}")

        os.makedirs(task_checkpoint, exist_ok = True)

        self.export_name(f"{task_checkpoint}/name", task.task_name)
        self.export_name(f"{task_checkpoint}/dual", task.task_name)
        
        with FileUtils.for_writing(f"{task_checkpoint}/labels") as file:
            for label in labels:
                file.write(f"{label}\n")
        
        with FileUtils.for_writing(f"{task_checkpoint}/weights") as file:
            file.write(f"# {numpy_weights.shape[0]} {numpy_weights.shape[1]}\n")
            for x in numpy_weights:
                for y in x:
                    file.write(f"{y} ")
                file.write("\n")
        
        with FileUtils.for_writing(f"{task_checkpoint}/biases") as file:
            file.write(f"# {numpy_bias.shape[0]}\n")
            for x in numpy_bias:
                file.write(f"{x} ")
            file.write("\n")
    
    def export_name(self, file_name: str, name: str) -> None:
        with open(file_name, "w", encoding=self.encoding) as file:
            file.write(f"{name}\n")

    def export_dual(self, file_name: str, dual_mode; bool) -> None:
        with open(file_name, "w", encoding=self.encoding) as file:
            if dual_mode:
                file.write("1\n")
            else:
                file.write("0\n")

    def export_encoder(self, checkpoint: str, tokenizer: AutoTokenizer, export_device: str) -> None:
        orig_words = ["Using", "transformers", "with", "ONNX", "runtime"]
        token_input = tokenizer(orig_words, is_split_into_words = True, return_tensors = "pt")
        # print(token_input)
        token_ids = token_input[names.INPUT_IDS].to(export_device) 
        
        inputs = (token_ids) 
        input_names = ["token_ids"] 
        output_names = ["sequence_output"]
        
        torch.onnx.export(
            self.encoder,
            inputs,
            checkpoint,
            export_params=True,
            do_constant_folding=True,
            input_names = input_names,
            output_names = output_names,
            opset_version=13, # see: https://chadrick-kwag.net/error-fix-onnxruntime-type-error-type-tensorint64-of-input-parameter-of-operatormin-in-node-is-invalid/
            dynamic_axes = {"token_ids": {1: "sent length"}}
        )

    # exports model in a format friendly for ingestion on the JVM
    def export_model(self, tasks: list[Task], tokenizer: AutoTokenizer, checkpoint_dir: str) -> None:
        # send the entire model to CPU for this export
        export_device = "cpu"
        self.to(export_device)

        # save the weights/bias in each linear layer
        task_folder = f"{checkpoint_dir}/tasks"
        os.makedirs(task_folder, exist_ok = True)
        for index, task in enumerate(tasks):
            task_checkpoint = f"{task_folder}/{index}"
            self.export_task(self.output_heads[str(task.task_id)], task, task_checkpoint)
    
        # save the encoder as an ONNX model
        onnx_checkpoint = f"{checkpoint_dir}/encoder.onnx"
        self.export_encoder(onnx_checkpoint, tokenizer, export_device)
        self.export_name(f"{checkpoint_dir}/encoder.name", parameters.transformer_name)
        

class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, task_id, dual_mode: bool=False, dropout_p: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.dual_mode = dual_mode
        self.classifier = nn.Linear(
            hidden_size if not self.dual_mode else hidden_size * 2, 
            num_labels
        )
        self.num_labels = num_labels
        self.task_id = task_id
        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        # torch.nn.init.xavier_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()    
            
    def summarize(self, task_id):
        print(f"Task {self.task_id} with {self.num_labels} labels.")
        print(f"Dropout is {self.dropout}")
        print(f"Classifier layer is {self.classifier}")

    def concatenate(self, sequence_output, head_positions):
      #print(f"in concat. sequence_output.size = {sequence_output.size()}; head_positions.size = {head_positions.size()}")
      long_head_positions = head_positions.to(torch.long)
      #print(f"head_positions: {long_head_positions}")
      head_states = sequence_output[torch.arange(sequence_output.shape[0]).unsqueeze(1), long_head_positions]
      #print(f"head_states.size = {head_states.size()}")
      # Concatenate the hidden states from modifier + head.
      modifier_head_states = torch.cat([sequence_output, head_states], dim=2)
      #print(f"modifier_head_states.size = {modifier_head_states.size()}")
      return modifier_head_states

    def forward(self, sequence_output, pooled_output, head_positions, labels=None, attention_mask=None, **kwargs):
        #print(f"sequence_output size = {sequence_output.size()}")
        sequence_output_for_classification = sequence_output if not self.dual_mode else self.concatenate(sequence_output, head_positions)

        sequence_output_dropout = self.dropout(sequence_output_for_classification)
        logits = self.classifier(sequence_output_dropout)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()            
            inputs = logits.view(-1, self.num_labels)
            targets = labels.view(-1)
            loss = loss_fn(inputs, targets)

        return logits, loss
