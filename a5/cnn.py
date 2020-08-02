#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5, stride: int = 1):
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels, out_channels,
                                    kernel_size, stride, padding=1)

    def forward(self, input: torch.Tensor):
        x_conv = self.conv_layer(input)
        x_conv_relu = torch.relu(x_conv)
        x_conv_out, _ = x_conv_relu.max(dim=-1)
        return x_conv_out
