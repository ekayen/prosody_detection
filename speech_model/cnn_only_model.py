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
from evaluate import evaluate


text_data = '../data/utterances.txt'
#speech_data = '../data/utterances_feats.pkl'
speech_data = '../data/cmvn_tensors.pkl'
labels_data = '../data/utterances_labels.pkl'

train_per = 0.6
dev_per = 0.2
print_every = 10
eval_every = 100
VERBOSE = False
STEPTHRU = False

train_params = {'batch_size': 8,
                     'shuffle': True,
                     'num_workers': 6}

eval_params = {'batch_size': 1,
                          'shuffle': True,
                          'num_workers': 6}
epochs = 1
pad_len = 200
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
                                  nn.Hardtanh(inplace=True),
                                  nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=self.kernel2, stride=self.stride2,
                                            padding=self.padding),
                                  nn.BatchNorm2d(self.out_channels),
                                  nn.Hardtanh(inplace=True)
                                  )

        # NO RNN VERSION:
        self.maxpool = nn.MaxPool1d(2)
        self.intermediate_fc_size = 300
        self.cnn_output_size = math.floor((self.seq_len - self.kernel1[0] + self.padding[0]*2)/self.stride1[0]) + 1
        self.cnn_output_size = math.floor((self.cnn_output_size - self.kernel2[0] + self.padding[0]*2)/self.stride2[0]) + 1
        self.cnn_output_size = int(((self.cnn_output_size)*self.out_channels)/2)
        self.fc1_no_rnn = nn.Linear(self.cnn_output_size, self.intermediate_fc_size)
        self.relu = nn.ReLU()
        self.fc2_no_rnn = nn.Linear(self.intermediate_fc_size,self.num_classes,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.inference_softmax = nn.Softmax(dim=-1)

    def forward(self,x,hidden):
        if VERBOSE: print('Input dims: ', x.view(x.shape[0], 1, x.shape[1], x.shape[2]).shape)
        x = self.conv(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))
        if VERBOSE: print('Dims after conv: ',x.shape)
        x = self.maxpool(x.view(x.shape[0],x.shape[1],x.shape[2]))
        if VERBOSE: print('Dims after pooling: ',x.shape)
        x = self.fc1_no_rnn(x.view(x.shape[0],x.shape[1]*x.shape[2]))
        x = self.relu(x)
        if STEPTHRU: import pdb;pdb.set_trace()
        if VERBOSE: print('Dims after fc1:', x.shape)
        x = self.fc2_no_rnn(x)
        if STEPTHRU: import pdb;pdb.set_trace()
        if VERBOSE: print('Dims after fc2:', x.shape)
        x = self.sigmoid(x)
        if STEPTHRU: import pdb;pdb.set_trace()
        if VERBOSE: print('Dims after sigmoid',x.shape)
        #x = self.inference_softmax(x)
        if VERBOSE: print('Dims after softmax:', x.shape)
        if STEPTHRU: import pdb;pdb.set_trace()
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

#evaluate(devset,eval_params,model)

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
            print('output: ', output[:,1:].squeeze())
            print('true labels: ',labels.float())
        loss = criterion(output[:, 1:].squeeze(), labels.float())  # No RNN
        loss.backward()
        optimizer.step()
        recent_losses.append(loss.detach())
        if len(recent_losses) > 50:
            recent_losses = recent_losses[1:]

        if timestep % print_every == 1:
            print('Train loss: ',sum(recent_losses)/len(recent_losses))
            process = psutil.Process(os.getpid())
            print('Memory usage at timestep ', timestep, ':', process.memory_info().rss / 1000000000, 'GB')
        if timestep % eval_every == 1 and not timestep==1:
            evaluate(devset,eval_params,model)
        timestep += 1

print('done')

process = psutil.Process(os.getpid())
print('Memory usage:',process.memory_info().rss/1000000000, 'GB')
