"""
Based very lightly on https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
by Sean Naren

Directly copied functions noted.

Modified by: Elizabeth Nielsen
"""

import pickle
from torch import nn
from torch.utils import data
import torch
import psutil
import os
from utils import UttDataset,plot_grad_flow,SynthDataset,plot_results
from evaluate import evaluate,logit_evaluate
import matplotlib.pyplot as plt
import random
import numpy as np
import math
random.seed(123)

model_name = 'rnn_fc'

text_data = '../data/utterances.txt'
speech_data = '../../gendata/synth_int_feats.pkl'
labels_data = '../../gendata/synth_int_labels.pkl'

train_per = 0.6
dev_per = 0.2
print_every = 1000
eval_every = 1000
VERBOSE = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_params = {'batch_size': 1,
                     'shuffle': True,
                     'num_workers': 6}

eval_params = {'batch_size': 1,
                          'shuffle': True,
                          'num_workers': 6}
epochs = 1
learning_rate = 0.001
LSTM_LAYERS = 1

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



        # TAKE OUT ALL LAYERS BUT FC FOR FIRST SYNTH TRIAL
        # RNN VERSION:
        #rnn_input_size = self.out_channels # This is what I think the input size of the LSTM should be -- channels, not time dim
        rnn_input_size = 1 # analogous to emb size
        self.lstm = nn.LSTM(input_size=rnn_input_size,
                            #batch_first=True,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.lin_input_size = self.hidden_size * 2
        else:
            self.lin_input_size = self.hidden_size

        self.fc = nn.Linear(self.lin_input_size,num_classes)


        self.inference_softmax = nn.Softmax(dim=-1)

    def forward(self,x,hidden):
        if VERBOSE: print('Input dims: ', x.shape)
        x = x.transpose(0,1)
        """
        if VERBOSE: print('Input dims: ', x.view(x.shape[0], 1, x.shape[1], x.shape[2]).shape)
        x = self.conv(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))
        if VERBOSE: print('Dims after conv: ',x.shape)
        x = x.view(x.shape[0],x.shape[1],x.shape[2]).transpose(1,2).transpose(0,1).contiguous()
        """
        if VERBOSE: print('Dims going into lstm: ',x.view(x.shape[0],self.batch_size,x.shape[1]).shape)
        x,hidden = self.lstm(x.view(x.shape[0],self.batch_size,x.shape[1]),hidden)
        if VERBOSE: print('Dims after lstm:', x.shape)
        x = x[-1,:,:] # TAKE LAST TIMESTEP
        if VERBOSE: print('Dims after slicing:', x.shape)
        x = self.fc(x.view(1,self.lin_input_size))
        #x = self.sigmoid(x)
        if VERBOSE: print('Dims after sigmoid',x.shape)
        #x = self.inference_softmax(x)
        if VERBOSE: print('Dims after softmax:', x.shape)
        return x,hidden


    def init_hidden(self,batch_size):
        if self.bidirectional:
            h0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(device)
            c0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(device)
        else:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(device)

        return (h0,c0)




print('Loading data ...')
with open(labels_data,'rb') as f:
    labels_dict = pickle.load(f)
with open(speech_data,'rb') as f:
    feat_dict = pickle.load(f)

#all_ids = list(labels_dict.keys())
all_ids = list(feat_dict.keys())
random.shuffle(all_ids)

train_ids = all_ids[:int(len(all_ids)*train_per)]
dev_ids = all_ids[int(len(all_ids)*train_per):int(len(all_ids)*(train_per+dev_per))]
test_ids = all_ids[int(len(all_ids)*(train_per+dev_per)):]

trainset = SynthDataset(train_ids,feat_dict,labels_dict)
devset = SynthDataset(dev_ids,feat_dict,labels_dict)

seq_len = 20

traingen = data.DataLoader(trainset, **train_params)

print('done')
print('Building model ...')

model = SpeechEncoder(seq_len=seq_len,
                      batch_size=train_params['batch_size'],
                      lstm_layers=LSTM_LAYERS,
                      hidden_size=128, # TODO figure out if this size will work
                      bidirectional=False,
                      num_classes=1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
recent_losses = []
timestep = 0

print('done')

print('Baseline eval....')

evaluate(devset,eval_params,model,device)

print('done')



plt_time = []
plt_losses = []
plt_acc = []
plt_dummy = []

print('Training model ...')
for epoch in range(epochs):
    for batch, labels in traingen: # TODO to generalize past binary classification, maybe change labels into one-hot
        batch, labels = batch.to(device), labels.to(device)
        model.zero_grad()
        hidden = model.init_hidden(train_params['batch_size'])
        output,_ = model(batch,hidden)
        if VERBOSE:
            print('output shape: ',output.shape)
            print('labels shape: ',labels.shape)
            print('output: ',output.squeeze())
            print('true labels: ',labels.float())



        loss = criterion(output.view(1),labels.float()) # With RNN
        loss.backward()
        plot_grad_flow(model.named_parameters())
        optimizer.step()
        recent_losses.append(loss.detach())

        if len(recent_losses) > 50:
            recent_losses = recent_losses[1:]

        if timestep % print_every == 1:
            plt_time.append(timestep)
            train_loss = (sum(recent_losses)/len(recent_losses)).item()
            print('Train loss: ',train_loss)
            plt_losses.append(train_loss)
            plt_dummy.append(0)
            process = psutil.Process(os.getpid())
            #print('Memory usage at timestep ', timestep, ':', process.memory_info().rss / 1000000000, 'GB')
            curr_pred_ones = 0
            curr_preds = 0
            #plt.show()
            plt_acc.append(logit_evaluate(devset,eval_params,model,device))
        timestep += 1
        #import pdb;pdb.set_trace()

print('done')

plot_results(plt_losses, plt_dummy, plt_acc, plt_time,model_name)

process = psutil.Process(os.getpid())
print('Memory usage:',process.memory_info().rss/1000000000, 'GB')
