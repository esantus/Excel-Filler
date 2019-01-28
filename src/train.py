#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 15:01:53 2018

@author: esantus
"""

from utilities import *

import pdb

import tqdm


def train_model(train_data, dev_data, class_balance, model, opt):
    '''
    Train model on the training and tune it on the dev set.
    
    If model does not improve dev performance within patience
    epochs, best model is restored and the learning rate halved
    to continue training.

    At the end of training, the function will restore the best model
    on the dev set.

        :param train_data: preprocessed data
        :param dev_data: preprocessed data
        :param models: models to be used for text classification
        :param opt: hyperparameters
        
        :return epoch_stats: a dictionary of metrics for train and dev
        :return model: best model
    '''
    
    snapshot = opt.model_full_path
    metrics_file_name = opt.output_file.split('.')[0] + ".txt"

    if opt.gpu:
        model = model.cuda()

    opt.lr = opt.init_lr
    optimizer = get_optimizer([model], opt)

    num_epoch_sans_improvement = 0
    epoch_stats = init_metrics_dictionary(modes=['train', 'dev'])
    step = 0
    tuning_key = "dev_{}".format(opt.tuning_metric)
    best_epoch_func = min if tuning_key == 'loss' else max

    train_loader = get_train_loader(train_data, opt, class_balance)
    dev_loader = get_dev_loader(dev_data, opt)
    
    metrics_file = open(metrics_file_name, 'a')

    # For every epoch...
    for epoch in range(1, opt.epochs + 1):
        metrics_file.write("-------------\nEpoch {}:\n".format(epoch))
        print("-------------\nEpoch {}:\n".format(epoch))
        
        # Load the training and dev sets...
        for mode, dataset, loader in [('Train', train_data, train_loader),
                                      ('Dev', dev_data, dev_loader)]:
            
            #try:
            
                train_model = mode == 'Train'
                metrics_file.write('{}\n'.format(mode))
                print('{}'.format(mode))
                key_prefix = mode.lower()
                epoch_details, step, _, _, _ = run_epoch(data_loader=loader, train_model=train_model, model=model,
                                                         optimizer=optimizer, step=step, opt=opt)
                
                epoch_stats, log_statement = collate_epoch_stat(epoch_stats, epoch_details, key_prefix, opt)
                
                # Log performance
                metrics_file.write('{}\n'.format(log_statement))
                print(log_statement)
            #except:
            #   pdb.set_trace()

        # Save model if beats best dev
        best_func = min if opt.tuning_metric == 'loss' else max
        if best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(opt.save_dir):
                os.makedirs(opt.save_dir)
            # Subtract one because epoch is 1-indexed and arr is 0-indexed
            epoch_stats['best_epoch'] = epoch - 1
            torch.save(model, snapshot)
        else:
            num_epoch_sans_improvement += 1

        if not train_model:
            metrics_file.write('---- Best Dev {} is {:.4f} at epoch {}\n\n'.format(
                opt.tuning_metric, epoch_stats[tuning_key][epoch_stats['best_epoch']],
                epoch_stats['best_epoch'] + 1))
            print('---- Best Dev {} is {:.4f} at epoch {}'.format(
                opt.tuning_metric, epoch_stats[tuning_key][epoch_stats['best_epoch']],
                epoch_stats['best_epoch'] + 1))

        # If the number of epochs without improvements is high, reduce the learning rate
        if num_epoch_sans_improvement >= opt.patience:
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            model.cpu()
            model = torch.load(snapshot)

            if opt.gpu:
                model = model.cuda()
            opt.lr *= .5
            optimizer = get_optimizer([model], opt)

    # Restore model to best dev performance
    if os.path.exists(opt.model_path):
        model.cpu()
        model = torch.load(snapshot)

    return epoch_stats, model
