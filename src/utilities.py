#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 15:00:42 2018

@author: esantus
"""

# Train the model
import sklearn.metrics
import sys, os

import numpy as np
import pandas as pd

import tqdm

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import csv

import pdb



def get_optimizer(models, opt):
    '''
    Save the parameters of every model in models and pass them to
    Adam optimizer.
    
        :param models: list of models (such as TextCNN, etc.)
        :param opt: arguments
        
        :return: torch optimizer over models
    '''
    params = []
    for model in models:
        params.extend([param for param in model.parameters() if param.requires_grad])
    return torch.optim.Adam(params, lr=opt.lr,  weight_decay=opt.weight_decay)


def init_metrics_dictionary(modes):
    '''
    Create dictionary with empty array for each metric in each mode
    
        :param modes: list with either train, dev or test
        
        :return epoch_stats: statistics for a given epoch
    '''
    epoch_stats = {}
    metrics = ['loss', 'obj_loss', 'k_selection_loss', 'k_continuity_loss',
               'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'mse']
    for metric in metrics:
        for mode in modes:
            key = "{}_{}".format(mode, metric)
            epoch_stats[key] = []
    return epoch_stats


def get_train_loader(train_data, opt, weights):
    '''
    Iterative train loader with sampler and replacer if class_balance
    is true, normal otherwise.
    
        :param train_data: training data
        :param opt: arguments
        
        :return train_loader: iterable training set
    '''
    
    if opt.class_balance:
        # If the class_balance is true: sample and replace
        sampler = data.sampler.WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_data),
                replacement=True)
        train_loader = data.DataLoader(
                train_data,
                num_workers=opt.num_workers,
                sampler=sampler,
                batch_size=opt.batch_size)
    else:
        # If the class_balance is false, do not sample
        train_loader = data.DataLoader(
            train_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            drop_last=False)
    return train_loader


def get_dev_loader(dev_data, opt):
    '''
    Iterative dev loader
    
        :param dev_data: dev set
        :param opt: arguments
        
        :return dev_loader: iterative dev set
    '''
    
    dev_loader = data.DataLoader(
        dev_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        drop_last=False)
    return dev_loader


def get_x_indx(batch, eval_model):
    '''
    Given a batch, return all the x
    
        :param batch: batch of dictionaries
        :param eval_model: true or false, for volatile
        
        :return x_indx: tensor of batch*x
    '''
    
    x_indx = autograd.Variable(batch['x'], volatile=eval_model)
    return x_indx


def get_char_x_indx(batch, eval_model):
    '''
    Given a batch, return all the char x
    
        :param batch: batch of dictionaries
        :param eval_model: true or false, for volatile
        
        :return char_x_indx: tensor of batch*char_x
    '''
    
    char_x_indx = autograd.Variable(batch['char_x'], volatile=eval_model)
    return char_x_indx


def get_loss(logit, y, opt):
    '''
    Return the cross entropy or mse loss
    
        :param logit: predictions
        :param y: gold standard
        :param opt: arguments
        
        :return loss: loss
    '''
    
    if opt.objective == 'cross_entropy':
        loss = F.cross_entropy(logit, y)
    elif opt.objective == 'mse':
        loss = F.mse_loss(logit, y.float())
    else:
        raise Exception("Objective {} not supported!".format(opt.objective))
    return loss


def tensor_to_numpy(tensor):
    '''
    Return a numpy matrix from a tensor

        :param tensor: tensor
        
        :return numpy_matrix: numpy matrix
    '''
    return tensor.data[0]


def get_metrics(preds, golds, opt):
    '''
    Return the metrics given predictions and golds
    
        :param preds: list of predictions
        :param golds: list of golds
        :param opt: arguments
        
        :return metrics: metrics dictionary
    '''
    metrics = {}

    if opt.objective  in ['cross_entropy', 'margin']:
        metrics['accuracy'] = sklearn.metrics.accuracy_score(y_true=golds, y_pred=preds)
        metrics['confusion_matrix'] = sklearn.metrics.confusion_matrix(y_true=golds,y_pred=preds)
        metrics['precision'] = sklearn.metrics.precision_score(y_true=golds, y_pred=preds, average="weighted")
        metrics['recall'] = sklearn.metrics.recall_score(y_true=golds,y_pred=preds, average="weighted")
        metrics['f1'] = sklearn.metrics.f1_score(y_true=golds,y_pred=preds, average="weighted")
        metrics['mse'] = "NA"
    elif opt.objective == 'mse':
        metrics['mse'] = sklearn.metrics.mean_squared_error(y_true=golds, y_pred=preds)
        metrics['confusion_matrix'] = "NA"
        metrics['accuracy'] = "NA"
        metrics['precision'] = "NA"
        metrics['recall'] = "NA"
        metrics['f1'] = 'NA'
    return metrics


def collate_epoch_stat(stat_dict, epoch_details, mode, opt):
    '''
    Update stat_dict with details from epoch_details and create
    log statement

        :param stat_dict: a dictionary of statistics lists to update
        :param epoch_details: list of statistics for a given epoch
        :param mode: train, dev or test
        :param opt: model run configuration

        :return stat_dict: updated stat_dict with epoch details
        :return log_statement: log statement sumarizing new epoch

    '''
    log_statement_details = ''
    for metric in epoch_details:
        loss = epoch_details[metric]
        stat_dict['{}_{}'.format(mode, metric)].append(loss)

        log_statement_details += ' -{}: {}'.format(metric, loss)

    log_statement = '\n {} - {}\n--'.format(opt.objective, log_statement_details )

    return stat_dict, log_statement



# Run each epoch
def run_epoch(data_loader, train_model, model, optimizer, step, opt, indx_to_class=False):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    
        :param data_loader: iterable dataset
        :param train_model: true if training, false otherwise
        :param model: text classifier, such as TextCNN
        :param optimizer: Adam
        :param opt: arguments
        
        :return epoch_stat:
        :return step: number of steps
        :return losses: list of losses
        :return preds: list of predictions
        :return golds: list of gold standards
    '''
    
    eval_model = not train_model
    data_iter = data_loader.__iter__()

    losses = []
    obj_losses = []
    
    preds = []
    golds = []
    texts = {}

    if train_model:
        model.train()
    else:
        model.eval()

    num_batches_per_epoch = len(data_iter)
    if train_model:
        num_batches_per_epoch = min(len(data_iter), 10000)

    for _ in tqdm.tqdm(range(num_batches_per_epoch)):
        # Get the batch
        batch = data_iter.next()
        
        if train_model:
            step += 1

        # Load X and Y
        x_indx = get_x_indx(batch, eval_model)
        char_x_indx = get_char_x_indx(batch, eval_model)
        
        y = autograd.Variable(batch['y'], volatile=eval_model)

        if opt.cuda:
            x_indx, y = x_indx.cuda(), y.cuda()

        if train_model:
            optimizer.zero_grad()

        logit, _ = model(x_indx, char_x_indx)

        if not opt.pr:
            # Calculate the loss
            loss = get_loss(logit, y, opt)
            obj_loss = loss

        # Backward step
        if train_model:
            loss.backward()
            optimizer.step()

        if not opt.pr:
            # Saving loss
            obj_losses.append(tensor_to_numpy(obj_loss))
            losses.append(tensor_to_numpy(loss))
        else:
            obj_losses = 0
            losses = []
        
        # Softmax, preds, text and gold
        batch_softmax = F.softmax(logit, dim=-1).cpu()
        golds.extend(batch['y'].numpy())
        preds.extend(torch.max(batch_softmax, 1)[1].view(y.size()).data.numpy())

        if opt.pr:
            text = batch['cols_vals']
            for k in text:
                if k not in texts:
                    texts[k] = []
                texts[k].extend(text[k])
                    
    # Get metrics
    if opt.pr:
        epoch_stat = {}
    else:
        epoch_metrics = get_metrics(preds, golds, opt)
        epoch_stat = {'loss' : np.mean(losses), 'obj_loss': np.mean(obj_losses)}
        for metric_k in epoch_metrics.keys():
            epoch_stat[metric_k] = epoch_metrics[metric_k]
        
    
    if opt.pr:
        texts['preds'] = [opt.y2label[preds[j]] for j in range(len(preds))]
        texts = pd.DataFrame(texts)
        writer = pd.ExcelWriter(opt.output_file)
        texts.to_excel(writer)

    return epoch_stat, step, losses, preds, golds
