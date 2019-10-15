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

text_data = '../data/utterances.txt'
speech_data = '../data/utterances_feats.pkl'
labels_data = '../data/utterances_labels.pkl'

train_per = 0.6
dev_per = 0.2

dataloader_params = {'batch_size': 32,
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

        self.mp = nn.MaxPool1d(self.seq_len,stride=self.seq_len)

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.batch_size),
            nn.Linear(hidden_size*2, self.num_classes, bias=False)
        )
        self.fc = fully_connected

        """
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        """
        self.inference_softmax = nn.Softmax(dim=-1)

    def forward(self,x,hidden):
        print('Input dims: ', x.view(x.shape[0], 1, x.shape[1], x.shape[2]).shape)
        x = self.conv(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))
        print('Dims after conv: ',x.shape)
        x = x.view(x.shape[0],x.shape[1],x.shape[2]).transpose(1,2).transpose(0,1).contiguous()
        print('Dims going into lstm: ',x.shape)
        x,hidden = self.lstm(x.view(x.shape[0],x.shape[1],x.shape[2]),hidden)
        print('Dims after lstm:', x.shape)
        x = torch.max(x,dim=0).values
        print('Dims after maxpool:', x.shape)
        x = self.fc(x.view(1,self.batch_size,self.hidden_size*2))
        print('Dims after fc:', x.shape)
        x = self.inference_softmax(x)
        print('Dims after softmax:', x.shape)
        print(x)
        return x,hidden

    def init_hidden(self):
        if self.bidirectional:
            h0 = torch.zeros(self.lstm_layers*2, self.batch_size, self.hidden_size).requires_grad_()#.to(device)
            c0 = torch.zeros(self.lstm_layers*2, self.batch_size, self.hidden_size).requires_grad_()#.to(device)
        else:
            h0 = torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size).requires_grad_()#.to(device)
            c0 = torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size).requires_grad_()#.to(device)

        return (h0,c0)







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

traingen = data.DataLoader(trainset, **dataloader_params)


model = SpeechEncoder(seq_len=pad_len,
                      batch_size=dataloader_params['batch_size'],
                      lstm_layers=3,
                      #num_classes=2
                      num_classes=1
                      )
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for batch, labels in traingen:
        model.zero_grad()
        hidden = model.init_hidden()
        output,_ = model(batch,hidden)
        print('output shape: ',output.shape)
        print('labels shape: ',labels.shape)
        loss = criterion(output.squeeze(),labels.float()) # TODO make labels into onehot repr
        loss.backward()
        optimizer.step()
        import pdb;pdb.set_trace()


process = psutil.Process(os.getpid())
print('Memory usage:',process.memory_info().rss/1000000000, 'GB')
