#!/usr/bin/env python
# coding: utf-8

import numpy as np
import platform
import random
import torch

class Configuration:
    def __init__(self):
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
        self.epochs: int = 2
        self.batch_size: int = 32
        self.weight_decay: float = 0.01

        # which transformer to use
        self.transformer_name: str = "bert-base-cased" # 'xlm-roberta-base' # 'distilbert-base-cased'
        self.model_name: str = f'{self.transformer_name}-mtl'

config = Configuration()
