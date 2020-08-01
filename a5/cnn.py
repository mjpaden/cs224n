#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5, stride: int = 1):
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels, out_channels,
                                    kernel_size, stride, padding=1)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        # input batch_size x char_embed_size x max_word_length
        x_conv = self.conv_layer(input)
        # x_conv batch_size x word_embed_size x max_word_length_prime
        x_conv_relu = self.relu(x_conv)
        # x_conv batch_size x word_embed_size x max_word_length_prime
        x_conv_out, _ = x_conv_relu.max(-1)
        # x_conv_out batch_size x word_embed_size
        return x_conv_out
    ### END YOUR CODE
