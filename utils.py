"""
This module contains helper for cycle GAN training.

Author: Rayylia
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import itertools

from abc import ABC, abstractmethod


class _GetDataset(data.Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.transform  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])

    def __getitem__(self, index):
        item_path = self.path[index]
        pil_item = Image.open(item_path).convert("RGB")
        tensor_item = self.transform(pil_item)
        return tensor_item
    
    def __len__(self):
        return len(self.path)


def set_loader(path_A, path_B, batch_size):
    # photo: 7038
    train_A_path = glob.glob(path_A)
    # monet: 300
    train_B_path = glob.glob(path_B)
    set_A = _GetDataset(train_A_path)
    set_B = _GetDataset(train_B_path)
    loader_A = data.DataLoader(set_A, batch_size=batch_size, shuffle=True)
    loader_B = data.DataLoader(set_B, batch_size=batch_size, shuffle=True)
    return loader_A, loader_B

