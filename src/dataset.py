from abc import ABCMeta, abstractmethod, abstractproperty
import torch.utils.data as data
import torch

import numpy as np
import pandas as pd

import re
import random
import tqdm

import pdb


SEPARATORS = [' ', '  ', '+', '.', ',', '%', '-', '\\', '/', '(', ')', '[', ']',
              '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '=', ':']



class AbstractDataset(data.Dataset):
    '''
    Abstract class
    '''
    
    __metaclass__ = ABCMeta

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample





class Dataset(AbstractDataset):
    '''
    Dataset loader
    '''
    
    def __init__(self, word_to_indx, opt): #ds_path, input_columns, output_columns, word_to_indx, max_length=80, max_char_length=15, train_size=.6, dev_size=.2):
        '''
        Load the dataset from Excel file

            :param word_to_indx: dictionary of word:index
            :param opt: arguments
            
            :return: nothing
        '''

        self.dataset = []
        self.class_balance = {}
        
        self.max_length = opt.max_words
        self.max_char_length = opt.max_chars
        
        self.word_to_indx = word_to_indx
        
        # Saving characters for char embeddings
        self.chars = list(set(opt.chars))
        self.char_to_indx = {c:i+1 for i, c in enumerate(self.chars)}
        self.char_to_indx['PADDING_CHAR'] = 0

        # Loading the Excel file
        ds = pd.read_excel(opt.excel_file, encoding = "ISO-8859-1")
        
        # Saving the classes
        self.classes = [x for x in ds[opt.output].unique() if str(x) != 'nan']
        if opt.pr:
            self.classes.append('nan')
        self.num_classes = len(self.classes)
        self.class_to_indx = {tag:i for i, tag in enumerate(self.classes)}
        self.indx_to_class = {self.class_to_indx[key]:key for key in self.class_to_indx}

        # Saving the datapoints
        for index, row in ds.iterrows():
            
            # Maximum number of entries to consider
            if pd.isnull(row[opt.output]) and opt.pr == False:
                continue
            
            # Turning the output string in a tokenized array
            output_string = str(row[opt.output]).strip()

            if output_string == '' and opt.pr == False:
                print("Current output column row is empty, passing to the next row")
                continue
            
            # Turning the input string in a tokenized array; every column is preceeded/followed by <s>
            if type(opt.input_columns) == list:
                input_string = ("<s> " + " <s> ".join([str(row[input_col]).strip() for input_col in opt.input_columns]) + " <s>").lower()
            else:
                input_string = ("<s> " + row[str(opt.input_columns)] + " <s>").lower()
            
            if len(set(input_string.split())) <= 1:
                print("Current input column row is empty, passing to the next row")
                continue
            
            input_string = self.tokenizeString(input_string, SEPARATORS)
            #print(input_string)
            
            # Creating x and y for every datapoint
            x = self.get_indices_tensor(input_string, self.word_to_indx, self.max_length)
            char_x = self.get_char_indices(input_string, self.char_to_indx, opt)
            y = self.class_to_indx[output_string]
            
            # Saving the class_balance
            if not y in self.class_balance:
                self.class_balance[y] = 0
            self.class_balance[y] += 1
            
            # Saving the dataset
            if type(opt.input_columns) == list:
                self.dataset.append({'cols_vals':{col:str(row[col]) for col in ds.columns}, 'input_fields':",".join([re.sub(",", " ", str(row[input_col]).strip()) for input_col in opt.input_columns]), 'text':input_string, 'x':x, 'char_x':char_x, 'y':y, 'label':output_string})
            else:
                self.dataset.append({'cols_vals':{col:str(row[col]) for col in ds.columns}, 'input_fields':str(row[opt.input_columns]).strip(), 'text':input_string, 'x':x, 'char_x':char_x, 'y':y, 'label':output_string})

        # Randomly split train in 60-20-20%
        if opt.train:
            random.shuffle(self.dataset)
            
        num_train = int(len(self.dataset)*opt.train_size)
        num_dev = int(len(self.dataset)*opt.dev_size)
        
        # Actual split
        self.train = self.dataset[:num_train]
        self.train_class_balance = self.calculate_weights(self.train)
        
        self.dev = self.dataset[num_train:num_train+num_dev]
        self.test = self.dataset[num_train+num_dev:]
        
        
    def calculate_weights(self, ds):
        count = [0] * self.num_classes
        weight_per_class = [0.0] * self.num_classes
        weight = [0] * len(ds)

        for item in ds:
            count[item['y']] += 1

        N= float(sum(count))

        for i in range(self.num_classes):
            try:
                weight_per_class[i] = N/float(count[i])
            except:
                weight_per_class[i] = N-1

        for indx, item in enumerate(ds):
            weight[indx] = weight_per_class[item['y']]

        return weight



    def tokenizeString(self, aString, separators):
        '''
        Given a string, it returns the tokenized string, saving all non-whitespace
        separators.
        '''
        
        do_not_save = [' ', '  ', '\t']
    
        separators.sort(key=len)
        tokens = []
        i = 0
        
        while i < len(aString):
            theSeparator = ''
            for current in separators:
                if current == aString[i:i+len(current)]:
                    theSeparator = current
            if theSeparator != "":
                tokens.append(theSeparator)
                i = i + len(theSeparator)
            else:
                if tokens == []:
                    tokens = ['']
                if(tokens[-1] in separators):
                    tokens += ['']
                tokens[-1] += aString[i]
                i += 1
        return [token for token in tokens if token not in do_not_save]
    
    
    
    def get_indices_tensor(self, text_arr, word_to_indx, max_length):
        '''
        Return a tensor of max_length with the word indices
        
            :param text_arr: text array
            :param word_to_indx: dictionary word:index
            :param max_length: maximum length of returned tensors
            
            :return x: tensor containing the indices
        '''
        
        pad_indx = 0
        text_indx = [word_to_indx[x] if x in word_to_indx else pad_indx for x in text_arr][:max_length]
        
        # Padding
        if len(text_indx) < max_length:
            text_indx.extend([pad_indx for _ in range(max_length - len(text_indx))])

        x =  torch.LongTensor(text_indx)

        return x
    
    
    
    def get_char_indices(self, text_arr, char_to_indx, opt):
        '''
        Return a tensor of max_length * max_char_length with the char indices
        
            :param text_arr: text array
            :param char_to_indx: dictionary word:index
            :param max_length: maximum length of returned tensors
            :param max_char_length: maximum char length of the returned tensor
            
            :return: tensor of char embedding indices
        '''
        char_indices = []
        for i in range(opt.max_words):
            if i < len(text_arr):
                word = text_arr[i][:opt.max_chars]
                word_indx = [self.char_to_indx[c] if c in self.char_to_indx else self.char_to_indx['PADDING_CHAR'] for c in word]
            else:
                word_indx = []
                
            # Padding
            if len(word_indx) < opt.max_chars:
                word_indx.extend([self.char_to_indx['PADDING_CHAR'] for _ in range(opt.max_chars - len(word_indx))])
            char_indices.append(word_indx)
            
        return torch.LongTensor(char_indices)



