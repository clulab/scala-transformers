#!/usr/bin/env python
# coding: utf-8

import numpy as np
import platform
import random
import torch 

class Configuration:
    def __init__(self) -> None:
        # select device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif 'arm64' in platform.platform():
            device = torch.device('mps') # 'mps'
        else:
            device = torch.device('cpu')
        print(f'Using device: {device.type}') 
        self.use_mps_device: bool = str(device) == 'mps'
        self.device: torch.device = device

        # random seed
        seed = 1234
        if seed is not None:
            print(f'Setting all random seeds to: {seed}')
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.seed: int = seed

        # pytorch ignores this label in the loss
        self.ignore_index: int = -100

        # training settings
        self.epochs: int = 20
        self.batch_size: int = 2 # batch size of 8 works with both bert-large and bert-base
        self.weight_decay: float = 0.01

        # which transformer to use
        # see this page for other options: https://huggingface.co/google/bert_uncased_L-4_H-256_A-4
        self.transformer_name: str = "bert-base-cased" # "bert-large-cased" # "xlm-roberta-base" # "distilbert-base-cased"
        self.model_name: str = f"{self.transformer_name.replace('/', '-')}-mtl" 

        # for dependency parsing, this dataset column indicates where the positions of the heads are stored
        self.HEAD_POSITIONS = "head_positions"

configuration = Configuration()
