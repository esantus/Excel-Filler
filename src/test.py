#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 15:26:12 2018

@author: esantus
"""

from utilities import *


# Testing the model

def test_model(test_data, model, opt, indx_to_class):
    '''
    Run the model on test data, and return statistics,
    including loss and accuracy.
    
        :param test_data: test data
        :param model: a model, like TextCNN
        :param opt: arguments
        
        :return test_stats:
    '''
    if opt.gpu:
        model = model.cuda()
        
    metrics_file_name = opt.output_file.split('.')[0] + ".txt"
    metrics_file = open(metrics_file_name, 'a')

    # Loading the test data as iterable
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        drop_last=False)

    #if opt.pr:   
    #else:
    # The function is defined before
    test_stats = init_metrics_dictionary(modes=['test'])

    mode = 'Test'
    train_model = False
    key_prefix = mode.lower()
    
    metrics_file.write("-------------\nTest\n")
    print("-------------\nTest")
    #pdb.set_trace()
    epoch_details, _, losses, preds, golds = run_epoch(
        data_loader=test_loader,
        train_model=train_model,
        model=model,
        optimizer=None,
        step=None,
        opt=opt, indx_to_class=indx_to_class)

    test_stats, log_statement = collate_epoch_stat(test_stats, epoch_details, 'test', opt)
    test_stats['losses'] = losses
    test_stats['preds'] = preds
    test_stats['golds'] = golds

    metrics_file.write('{}\n'.format(log_statement))
    print(log_statement)

    return test_stats
