#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):

    def __init__(self, dim: int):
        super(Highway, self).__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_proj = torch.relu(self.proj(input))
        x_gate = torch.sigmoid(self.gate(input))
        output = x_gate * x_proj + (1 - x_gate) * input
        return output
