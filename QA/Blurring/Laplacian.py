import torch.nn as nn
import torch
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
import kornia
import openslide
import torchvision.transforms as transforms

class Laplacian(nn.Module):
    '''Args:
    kernel_size: int, the size of the kernel, must be odd number
    border_type: str ('constant', 'reflect', 'replicate' or 'circular'.), the padding mode to be applied before convolving.
    normalized: bool, if True, L1 norm of the kernel is set to 1.
    return_variance: bool, if True, return the variance of the laplacian matrix
    Input: Tensor, shape: (B, C, H, W)
    Output: Tensor, shape: (B, C, H, W)'''
    def __init__(self, kernel_size=11, border_type='reflect', normalized=False, return_variance=True):
        super().__init__()

        self.kernel_size: int = kernel_size
        self.border_type: str = border_type
        self.normalized: bool = normalized
        self.return_variance: bool = return_variance

    def laplacian(self, image: torch.Tensor):
        return kornia.filters.laplacian(image, self.kernel_size, self.border_type, self.normalized)

    def forward(self, image: torch.Tensor):
        if self.return_variance:
            return torch.var((self.laplacian(image)))
        else:
            return self.laplacian(image)


