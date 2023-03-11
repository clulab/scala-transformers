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
epochs = 20
batch_size = 8 # batch size of 8 works with both bert-large and bert-base
weight_decay = 0.01

# which transformer to use
# see this page for other options: https://huggingface.co/google/bert_uncased_L-4_H-256_A-4
transformer_name = 'roberta-base' # 'google/bert_uncased_L-4_H-512_A-8' # 'bert-base-cased' # 'bert-large-cased' # 'xlm-roberta-base' # 'distilbert-base-cased'
model_name = f'{transformer_name.replace("/", "-")}-mtl' 

# for dependency parsing, this dataset column indicates where the positions of the heads are stored
HEAD_POSITIONS = "head_positions"



