#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:20:03 2019

@author: esantus
"""

import pdb
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


# Encoder
class Encoder(nn.Module):
    '''
    Load the word and character embeddings and call the model
    '''

    def __init__(self, embeddings, opt):
        '''
        Load word and character embeddings and call the TextCNN model

            :param embeddings: tensor with word embeddings
            :param opt: configuration

            :return: nothing
        '''
        super(Encoder, self).__init__()

        self.opt = opt

        # Saving the parameters
        self.num_class = opt.num_classes

        self.filters = opt.filters

        self.dropout = opt.dropout

        self.len_chars = len(list(set(opt.chars))) + 1

        self.max_words = opt.max_words
        self.max_chars = opt.max_chars

        self.char_emb_dims = opt.char_emb_dims

        # Loading the word embeddings in the Neural Network
        word_vocab_size, word_hidden_dim = embeddings.shape
        self.emb_dims = word_hidden_dim
        self.emb_layer = nn.Embedding(word_vocab_size, word_hidden_dim)
        self.emb_layer.weight.data = torch.from_numpy(embeddings)
        self.emb_layer.weight.requires_grad = True

        if opt.char:
            # Loading the character embeddings in the Neural Network
            self.filter_num = opt.char_filter_num
            char_vocab_size = self.len_chars
            char_hidden_dim = self.char_emb_dims
            self.char_emb_dims = char_hidden_dim
            self.char_emb_layer = nn.Embedding(char_vocab_size, char_hidden_dim)
            self.char_emb_layer.weight.requires_grad = True

            # Call the char CNN
            self.char_emb = TextCNN(opt, opt.char_emb_dims, max_pool_over_time=True)
            # Calling the Classification Model, followed by a fully connected hidden layer
            self.cnn = TextCNN(opt, opt.emb_dims + opt.char_emb_dims, max_pool_over_time=True)
            self.emb_fc = nn.Linear(word_hidden_dim * 2, word_hidden_dim + char_hidden_dim)
        else:
            self.filter_num = opt.filter_num
            self.emb_fc = nn.Linear(word_hidden_dim, word_hidden_dim)
            # Calling the Classification Model, followed by a fully connected hidden layer
            self.cnn = TextCNN(opt, opt.emb_dims, max_pool_over_time=True)

        # The hidden fully connected layer size is given by the number of filters
        # times the filter size, by the number of hidden dimensions
        num_numbers = len(opt.input_numbers)
        print('num_numbers is',num_numbers)


        if opt.input_columns != []:
            self.fc = nn.Linear(len(self.filters) * self.filter_num+num_numbers, word_hidden_dim)
        else:
            self.fc = nn.Linear(num_numbers, word_hidden_dim)

        # Dropout and final layer
        self.dropout = nn.Dropout(self.dropout)
        self.hidden = nn.Linear(word_hidden_dim, self.num_class if opt.objective != 'mse' else 1)

    def forward(self, x_indx, char_x_indx = False):

        '''
        Forward step

            :param x_indx: batch of word indices

            :return logit: predictions
            :return: hidden layer
        '''

        if isinstance( x_indx,tuple):
            x_indx,num_x = x_indx

        if self.opt.debug:
            pdb.set_trace()

        if not isinstance(x_indx,list):
            word_x = self.emb_layer(x_indx.squeeze(1))

            if self.opt.char:
                char_x = self.char_emb(self.char_emb_layer(char_x_indx.view(-1, self.max_chars)).transpose(1, 2)).view(
                    -1,
                    self.max_words,
                    self.emb_dims)
                x = torch.cat((word_x, char_x), 2)
            else:
                x = word_x

            # Non linear projection with dropout
            x = F.relu(self.emb_fc(x))
            x = self.dropout(x)
            # TextNN, fully connected and non linearity
            x = torch.transpose(x, 1, 2)  # Transpose x dimensions into (Batch, Emb, Length)

            # Concatenate the char embeddings
            hidden = self.cnn(x)

            if not isinstance(num_x,list):
                hidden = torch.cat((hidden, num_x), 1)
        else:
            hidden = num_x



        hidden = F.relu(self.fc(hidden))

        # Dropout and final layer
        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)
        return logit, hidden


# Model
class TextCNN(nn.Module):
    '''
    CNN for Text Classification
    '''

    def __init__(self, opt, emb_dim, max_pool_over_time=False):
        '''
        Convolutional Neural Network

            :param num_layers: number of layers
            :param filters: filters shape
            :param filter_num: number of filters
            :param emb_dims: embedding dimensions
            :param max_pool_over_time: boolean

            :return: nothing
        '''
        super(TextCNN, self).__init__()

        # Saving the parameters
        self.num_layers = opt.num_layers
        self.filters = opt.filters
        self.filter_num = opt.filter_num
        self.emb_dims = emb_dim  # opt.emb_dims + opt.char_emb_dims
        self.gpu = opt.gpu
        self.max_pool = max_pool_over_time

        self.layers = []

        # For every layer...
        for l in range(self.num_layers):
            convs = []

            # For every filter...
            for f in self.filters:
                # Defining the sizes
                in_channels = self.emb_dims if l == 0 else self.filter_num * len(self.filters)
                kernel_size = f

                # Adding the convolutions in the list
                conv = nn.Conv1d(in_channels=in_channels, out_channels=self.filter_num, kernel_size=kernel_size)
                self.add_module('layer_' + str(l) + '_conv_' + str(f), conv)
                convs.append(conv)

            self.layers.append(convs)

    def _conv(self, x):
        '''
        Left padding and returning the activation

            :param x: input tensor (batch, emb, length)
            :return layer_activ: activation
        '''

        layer_activ = x

        for layer in self.layers:
            next_activ = []

            for conv in layer:
                # Setting the padding dimensions: it is like adding
                # kernel_size - 1 empty embeddings
                left_pad = conv.kernel_size[0] - 1
                pad_tensor_size = [d for d in layer_activ.size()]
                pad_tensor_size[2] = left_pad
                left_pad_tensor = autograd.Variable(torch.zeros(pad_tensor_size))

                if self.gpu:
                    left_pad_tensor = left_pad_tensor.cuda()

                # Concatenating the padding to the tensor
                padded_activ = torch.cat((left_pad_tensor, layer_activ), dim=2)

                # onvolution activation
                next_activ.append(conv(padded_activ))

            # Concatenating accross channels
            layer_activ = F.relu(torch.cat(next_activ, 1))
            # pdb.set_trace()
        return layer_activ

    def _pool(self, relu):
        '''
        Max Pool Over Time
        '''

        pool = F.max_pool1d(relu, relu.size(2)).squeeze(-1)
        return pool

    def forward(self, x):
        '''
        Forward steps over the x

            :param x: input (batch, emb, length)

            :return activ: activation
        '''
        # pdb.set_trace()
        activ = self._conv(x)

        # Pooling over time?
        if self.max_pool:
            activ = self._pool(activ)

        return activ
