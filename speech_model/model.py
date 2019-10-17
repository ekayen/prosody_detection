"""
Based on https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
by Sean Naren

Directly copied functions noted.

Modified by: Elizabeth Nielsen
"""

import pandas as pd
import pickle
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import torch
import psutil
import os
import time
from utils import UttDataset
import math
import random
random.seed(0)
import numpy as np

text_data = '../data/utterances.txt'
speech_data = '../data/utterances_feats.pkl'
labels_data = '../data/utterances_labels.pkl'

train_per = 0.6
dev_per = 0.2
print_every = 10
eval_every = 100
VERBOSE = False

train_params = {'batch_size': 16,
                     'shuffle': True,
                     'num_workers': 6}

eval_params = {'batch_size': 1,
                          'shuffle': True,
                          'num_workers': 6}
epochs = 1
pad_len = 750
learning_rate = 0.001

class SpeechEncoder(nn.Module):
    def __init__(self,
                 seq_len,
                 batch_size,
                 hidden_size=512,
                 bidirectional=True,
                 lstm_layers=3,
                 num_classes=2):
        super(SpeechEncoder,self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes

        self.feat_dim = 16
        self.in_channels = 1
        self.hidden_channels = 128
        self.out_channels = 256
        self.kernel1 = (9,self.feat_dim)
        self.kernel2 = (9,1)
        self.stride1 = (2,self.feat_dim)
        self.stride2 = (2,1)
        self.padding = (4,0)

        self.conv = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=self.kernel1, stride=self.stride1,
                                            padding=self.padding),
                                  nn.BatchNorm2d(self.hidden_channels),
                                  nn.ReLU(inplace=True), # TODO figure out if inplace is correct
                                  nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=self.kernel2, stride=self.stride2,
                                            padding=self.padding),
                                  nn.BatchNorm2d(self.out_channels),
                                  nn.ReLU(inplace=True)
                                  )

        """
        # This is what that implementation of deepspeech does, but to me it seems like the input size to the LSTM should be the number of channels, not the time dim

        rnn_input_size = self.seq_len
        rnn_input_size = math.ceil((rnn_input_size - self.kernel1[0] + 2 * self.padding[0]) / (self.stride1[0]))
        rnn_input_size = math.ceil((rnn_input_size - self.kernel2[0] + 2 * self.padding[0]) / (self.stride2[0]))
        print('RNN input size',rnn_input_size)
        """
        rnn_input_size = self.out_channels # This is what I think the input size of the LSTM should be -- channels, not time dim
        self.lstm = nn.LSTM(input_size=rnn_input_size,
                            #batch_first=True,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=self.bidirectional)

        # TODO figure out how to set the dim of batch norm layer so it works without prespecified batch size, or
        # TODO figure out how to turn it off at eval time

        # TODO OR actually maybe not? Taking batch norm out seems to have made the different examples in a batch decouple from each other, which is desirable

        if self.bidirectional:
            self.lin_input_size = self.hidden_size * 2
        else:
            self.lin_input_size = self.hidden_size

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.batch_size),
            nn.Linear(self.lin_input_size, self.num_classes, bias=False)
        )

        self.bn = nn.BatchNorm1d(self.batch_size)
        #self.fc = fully_connected
        self.fc = nn.Linear(self.lin_input_size, self.num_classes, bias=False)
        self.sigmoid = nn.Sigmoid()
        """
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        """
        self.inference_softmax = nn.Softmax(dim=-1)

    def forward(self,x,hidden):
        if VERBOSE: ('Input dims: ', x.view(x.shape[0], 1, x.shape[1], x.shape[2]).shape)
        x = self.conv(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))
        if VERBOSE: print('Dims after conv: ',x.shape)
        x = x.view(x.shape[0],x.shape[1],x.shape[2]).transpose(1,2).transpose(0,1).contiguous()
        if VERBOSE: print('Dims going into lstm: ',x.shape)
        x,hidden = self.lstm(x.view(x.shape[0],x.shape[1],x.shape[2]),hidden)
        if VERBOSE: print('Dims after lstm:', x.shape)
        #x = torch.max(x,dim=0).values # MAXPOOL OVER TIME DIM
        x = x[-1,:,:] # TAKE LAST TIMESTEP
        if VERBOSE: print('Dims after compression:', x.shape)
        #if x.shape[0] > 1: # TODO fix this again
        #    x = self.bn(x)
        x = self.fc(x.view(1,x.shape[0],self.lin_input_size))
        if VERBOSE: print('Dims after fc:', x.shape)
        x = self.sigmoid(x)
        if VERBOSE: print('Dims after sigmoid',x.shape)
        x = self.inference_softmax(x)
        if VERBOSE: print('Dims after softmax:', x.shape)
        return x,hidden

    def init_hidden(self,batch_size):
        if self.bidirectional:
            h0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_()#.to(device)
            c0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_()#.to(device)
        else:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_()#.to(device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_()#.to(device)

        return (h0,c0)




print('Loading data ...')
with open(labels_data,'rb') as f:
    labels_dict = pickle.load(f)
with open(speech_data,'rb') as f:
    feat_dict = pickle.load(f)

all_ids = list(labels_dict.keys())
random.shuffle(all_ids)

train_ids = all_ids[:int(len(all_ids)*train_per)]
dev_ids = all_ids[int(len(all_ids)*train_per):int(len(all_ids)*(train_per+dev_per))]
test_ids = all_ids[int(len(all_ids)*(train_per+dev_per)):]

trainset = UttDataset(train_ids,feat_dict,labels_dict,pad_len)
devset = UttDataset(dev_ids,feat_dict,labels_dict,pad_len)

traingen = data.DataLoader(trainset, **train_params)

print('done')
print('Building model ...')

model = SpeechEncoder(seq_len=pad_len,
                      batch_size=train_params['batch_size'],
                      lstm_layers=3,
                      bidirectional=False,
                      num_classes=2)
                      #num_classes=1)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
recent_losses = []
timestep = 0

print('done')

print('Baseline eval....')

def evaluate(dataset,dataloader_params):
    true_pos_pred = 0
    total_pred = 0
    dataloader = data.DataLoader(dataset, **dataloader_params)
    with torch.no_grad():
        for x,y in dataloader:
            hidden = model.init_hidden(dataloader_params['batch_size'])
            output,_ = model(x,hidden)
            output = np.argmax(output)
            total_pred += 1
            if output.item() == y.item():
                true_pos_pred += 1
    acc = true_pos_pred/total_pred
    print('Accuracy: ',acc)

evaluate(devset,eval_params)

print('done')

print('Training model ...')
for epoch in range(epochs):
    for batch, labels in traingen: # TODO to generalize past binary classification, maybe change labels into one-hot
        model.zero_grad()
        hidden = model.init_hidden(train_params['batch_size'])
        output,_ = model(batch,hidden)
        if VERBOSE:
            print('output shape: ',output.shape)
            print('labels shape: ',labels.shape)
            print('output: ',output[:,:,1:].squeeze())
            print('true labels: ',labels.float())
        loss = criterion(output[:,:,1:].squeeze(),labels.float()) # TODO make labels into onehot repr
        loss.backward()
        optimizer.step()
        recent_losses.append(loss.detach())
        if len(recent_losses) > 50:
            recent_losses = recent_losses[1:]

        if timestep % print_every == 1:
            print('Train loss: ',sum(recent_losses)/len(recent_losses))
        if timestep % eval_every == 1:
            evaluate(devset,eval_params)
        process = psutil.Process(os.getpid())
        print('Memory usage at timestep ',timestep,':', process.memory_info().rss / 1000000000, 'GB')
        timestep += 1
        #import pdb;pdb.set_trace()

print('done')

process = psutil.Process(os.getpid())
print('Memory usage:',process.memory_info().rss/1000000000, 'GB')
