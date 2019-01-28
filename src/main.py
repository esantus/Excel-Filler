#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:23:19 2019

@author: esantus
"""

import os
import sys

import pdb

import optparse

import torch
import embeddings as emb
import dataset as ds
import model as mdl
import train as train
import test as test

import pickle



def main():
    parser = optparse.OptionParser()
    parser.set_defaults(chars=' !?><=:;[]{}^@\"#$%&\'()*+,_-./abcdefghijklmnopqrstuvwyz1234567890',
                        save_dir = 'snapshot', model_full_path='', config='', output_file='', y2label={})
    
    parser.add_option('--mode', default='train', action='store', type=str, help='save the mode (train, test, predict)')

    parser.add_option('-d', '--debug', default=False, action='store_true', help='if True, print debug information')
    parser.add_option('-g', '--gpu', default=True, action='store_true', help='if True, use the GPU')
    parser.add_option('-b', '--class_balance', default=True, action='store_true', help='if True, use class balance')
    parser.add_option('-r', '--char', default=True, action='store_true', help='if True, use character embeddings too')
    
    parser.add_option('-f', '--excel_file', default='../datasets/LIMS.xlsx', action='store', help='excel file to be used for Training/Prediciton')
    parser.add_option('-e', '--embedding_file', default='../embeddings/glove.6B.300d.txt', action='store', help='embedding file to be used')
    parser.add_option('-i', '--input_columns', default='Product', action='store', type=str, help='input columns in format: x1,x2...,xN')
    parser.add_option('-o', '--output_columns', default='Assay', action='store', type=str, help='output columns in format: x1,x2...,xN')

    parser.add_option('-s', '--model_path', default='../snapshot/', action='store', type=str, help='folder in which the model is (going to be) saved')
    parser.add_option('-n', '--model_name', default='_model.pt', action='store', type=str, help='model name as it is (going to be) saved')
    parser.add_option('-m', '--tuning_metric', default='loss', action='store', type=str, help='tuning metric')
    parser.add_option('-j', '--objective', default='cross_entropy', action='store', type=str, help='objective function')
    
    parser.add_option('--init_lr', default=0.0001, action='store', type=float, help='save the initial learning rate')
    parser.add_option('--epochs', default=320, action='store', type=int, help='save the number of epochs')
    parser.add_option('--batch_size', default=16, action='store', type=int, help='save the batch size')
    parser.add_option('--patience', default=5, action='store', type=int, help='save the patience before cutting the learning rate')
    parser.add_option('--emb_dims', default=300, action='store', type=int, help='save the embedding dimension')
    parser.add_option('--char_emb_dims', default=300, action='store', type=int, help='save the char embedding dimension')
    parser.add_option('--hidden_dims', default=100, action='store', type=int, help='save the number of hidden dimensions for TextCNN')
    parser.add_option('--num_layers', default=1, action='store', type=int, help='save the number of layers')
    parser.add_option('--dropout', default=0.2, action='store', type=float, help='save the dropout probability')
    parser.add_option('--weight_decay', default=1e-3, action='store', type=float, help='save the weight decay')
    parser.add_option('--filter_num', default=100, action='store', type=int, help='save the number of filters')
    parser.add_option('--filters', default=[3, 4, 5], action='store', type=str, help='save the list of filters in format x1,x2...,xN')
    parser.add_option('--num_class', default=100, action='store', type=int, help='save the number of classes in the output')
    parser.add_option('--max_words', default=30, action='store', type=int, help='save the maximum number of words to use from the input')
    parser.add_option('--max_chars', default=10, action='store', type=int, help='save the maximum number of chars to use for every word')
    parser.add_option('--train_size', default=.6, action='store', type=float, help='save the relative size of the training set')
    parser.add_option('--dev_size', default=.2, action='store', type=float, help='save the relative size of the dev set')
    parser.add_option('--test_size', default=.2, action='store', type=float, help='save the relative size of the test set')
    parser.add_option('--num_workers', default=0, action='store', type=int, help='save the number of workers')
    
    (opt, args) = parser.parse_args()
    
    if opt.mode == 'train':
        opt.train = True
        opt.pr = False
    elif opt.mode == 'predict':
        opt.train = False
        opt.pr = True
    elif opt.mode == 'test':
        opt.train = False
        opt.pr = False
        
    opt.input_columns = opt.input_columns.split(',')
    opt.output_columns = opt.output_columns.split(',')
    if type(opt.filters) == str:
        opt.filters = [int(x) for x in opt.filters.split(',')]
    
    
    if opt.debug:
        print('Input columns: {}\nOutput columns: {}\nFilters: {}'.format(opt.input_columns, opt.output_columns, opt.filters))
    

    for output in opt.output_columns:
        opt.output = output
        opt.model_full_path = '{}'.format(os.path.join(opt.model_path, output + opt.model_name))
        opt.config = '{}'.format(os.path.join(opt.model_path, 'config_' + output + opt.model_name.split('.')[0] + '.pt'))
        opt.output_file = '{}'.format(os.path.join(opt.model_path, 'output_' + output + '_' + opt.excel_file.split('/')[-1].split('.')[0] + '.xlsx'))
        
        print('Model: {}\nConfig: {}\nOutput File: {}'.format(opt.model_full_path, opt.config, opt.output_file))

        # Loading the embeddings and the word index
        emb_tensor, word_to_indx = emb.load_embeddings(opt.embedding_file, opt.emb_dims)
        
        # Load the dataset in the format list({'x':WORD_INDEX_TENSOR, 'y':TAG_NUMBER,
        # 'text':'full text...', 'y_name':TAG_NAME}, {...}).
        dataset = ds.Dataset(word_to_indx, opt) 
	opt.num_classes = dataset.num_classes
        opt.y2label = dataset.indx_to_class
        print("Dataset: {}\nTrain: {}\nDev: {}\nTest: {}".format(len(dataset.dataset), len(dataset.train), len(dataset.dev), len(dataset.test)))

        if opt.debug:
            pdb.set_trace()
        
        if opt.train:
            # Create and train the model, save it and save the configuration file.
            config = pickle.dump(opt, open(opt.config, 'wb'))
            encoder = mdl.Encoder(emb_tensor, opt)
            train.train_model(dataset.train, dataset.dev, dataset.train_class_balance, encoder, opt) # Not necessary to send the index_to_class
            
            test.test_model(dataset.test, encoder, opt, opt.y2label)
            print("Train: {}\nDev: {}\nTest: {}".format(len(dataset.train), len(dataset.dev), len(dataset.test)))
        else:
            # Load the model and the configuration file.    
            args = pickle.load(open(opt.config, 'rb'))
            
            # Setting the correct combination
            args.train = False
            if opt.pr == True:
                args.pr = True
            opt = args
            
            encoder = torch.load(opt.model_full_path)
            
            test.test_model(dataset.dataset, encoder, opt, indx_to_class=opt.y2label) # Necessary to send the index_to_class
            print("Test set: {}".format(len(dataset.dataset)))


if __name__ == "__main__":
    main()


