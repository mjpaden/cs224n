#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        dropout_rate = 0.3
        n_chars = len(vocab.char2id)
        self.char_embed_size = 50
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.char_embed = nn.Embedding(n_chars, self.char_embed_size)
        self.conv = CNN(self.char_embed_size, word_embed_size)
        self.highway = Highway(word_embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        # sentence_length x batch_size x max_word_length
        orig_size = input.size()
        merged_input = input.reshape(-1, orig_size[-1])
        # s*b x max_word_length
        x_char_embed = self.char_embed(merged_input)
        # s*b x max_word_length x char_embed_size
        x_reshaped = x_char_embed.transpose(-1, -2)
        # s*b x char_embed_size x max_word_length
        x_conv = self.conv(x_reshaped)
        # out s*b x word_embed_size
        x_emb = self.highway(x_conv)
        x_drop = self.dropout(x_emb)
        split_output = x_drop.reshape(list(orig_size[:-1]) + [self.word_embed_size])
        return split_output
        ### END YOUR CODE
