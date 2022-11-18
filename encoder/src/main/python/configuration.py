#!/usr/bin/env python
# coding: utf-8

import platform
import random

import torch 
import numpy as np

# select device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif 'arm64' in platform.platform():
    device = torch.device('mps') # 'mps'
else:
    device = torch.device('cpu')
use_mps_device = True if str(device) == 'mps' else False
print(f'Using device: {device.type}') 

# random seed
seed = 1234

# set random seed
if seed is not None:
    print(f'Setting all random seeds to: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# pytorch ignores this label in the loss
ignore_index = -100

# training settings
epochs = 2
batch_size = 32
weight_decay = 0.01

# which transformer to use
transformer_name = "bert-base-cased" # 'xlm-roberta-base' # 'distilbert-base-cased'
model_name = f'{transformer_name}-mtl'



