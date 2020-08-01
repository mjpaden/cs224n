#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, dim: int):
        super(Highway, self).__init__()
        self.gate = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_gate = self.relu(self.gate(input))
        x_proj = self.sigmoid(self.proj(input))
        output = x_gate * x_proj + (1 - x_gate) * input
        return output

    ### END YOUR CODE