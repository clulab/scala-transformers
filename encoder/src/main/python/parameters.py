#!/usr/bin/env python
# coding: utf-8

import numpy as np
import platform
import random
import torch 

class Parameters:
    avoid_cuda = False
    avoid_mps = False
    # select device
    if not avoid_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not avoid_mps and "arm64" in platform.platform():
        device = torch.device("mps") # "mps"
    else:
        device = torch.device("cpu")
    print(f"Using device: {device.type}") 
    use_mps_device: bool = str(device) == "mps"
    use_cuda_device: bool = str(device) == "cuda"
    device: torch.device = device

    # random seed
    seed = 1234
    if seed is not None:
        print(f"Setting all random seeds to: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # pytorch ignores this label in the loss
    ignore_index: int = -100

    # training settings
    epochs: int = 20
    batch_size: int = 2 # batch size of 8 works with both bert-large and bert-base
    weight_decay: float = 0.01

    # which transformer to use
    # see this page for other options: https://huggingface.co/google/bert_uncased_L-4_H-256_A-4
    # transformer_name: str = "bert-base-cased" 
    # transformer_name: str = "distilbert-base-cased"
    transformer_name: str = "roberta-base" 
    # transformer_name: str = "xlm-roberta-base" 
    # transformer_name: str = "google/bert_uncased_L-4_H-512_A-8" 
    # transformer_name: str = "google/electra-small-discriminator"
    # transformer_name: str = "microsoft/deberta-v3-base" 

    # the encoding used by default for reading and writing files       
    encoding = "UTF-8"

    def get_model_name(transformer_name: str) -> str:
        return f"{transformer_name.replace('/', '-')}-mtl"

    model_name: str = get_model_name(transformer_name)
