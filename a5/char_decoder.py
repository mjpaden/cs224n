#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character
            embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language.
            See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = \
            nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = \
            nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                         padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM
            before reading the input characters. A tuple of two tensors
            of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length,
            batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state
            of the LSTM after reading the input characters. A tuple
            of two tensors of shape (1, batch, hidden_size)
        """
        input_emb = self.decoderCharEmb(input)
        lstm_outputs, dec_hidden = \
            self.charDecoder(input_emb, dec_hidden)
        scores = self.char_output_projection(lstm_outputs)
        return scores, dec_hidden

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length,
            batch_size). Note that "length" here and in forward() need
            not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state
            of the LSTM, obtained from the output of the word-level decoder.
            A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum*
            of cross-entropy losses of all the words in the batch.
        """
        char_pad = self.target_vocab.char_pad
        loss = nn.CrossEntropyLoss(ignore_index=char_pad, reduction='sum')

        scores, _ = self(char_sequence, dec_hidden)

        vocab_size = scores.size(2)
        scores = scores[:-1].reshape([-1, vocab_size])

        targets = char_sequence[1:].reshape([-1])

        train_loss = loss(scores, targets)
        return train_loss

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state
            of the LSTM, a tuple of two tensors of size
            (1, batch_size, hidden_size)
        @param device: torch.device
            (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size)
            of strings, each of which has length <= max_length.
            The decoded strings should NOT contain the start-of-word and
            end-of-word characters.
        """

        batch_size = initialStates[0].size(1)

        vocab = self.target_vocab
        sow_idx = vocab.start_of_word
        eow_idx = vocab.end_of_word

        next_idxs = torch.tensor([[sow_idx] * batch_size], device=device)
        states = initialStates
        words = []

        for i in range(max_length):
            scores, states = self(next_idxs, states)
            next_idxs = scores.argmax(dim=-1)
            words.append(next_idxs)

        words = torch.cat(words).T.tolist()
        decodedWords = []
        for w in words:
            decoded = ''
            for c in w:
                if c == eow_idx:
                    break
                else:
                    decoded += vocab.id2char[c]
            decodedWords.append(decoded)

        return decodedWords
