"""
Based on https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
by Sean Naren

Directly copied functions noted.

Modified by: Elizabeth Nielsen
"""

import matplotlib.pyplot as plt
import pickle
from torch import nn
from torch.utils import data
import torch
import psutil
import os
from utils import UttDataset,plot_grad_flow,SynthDataset,plot_results
import math
import random
import numpy as np
random.seed(123)
from evaluate import evaluate,logit_evaluate


text_data = '../data/utterances.txt'
#speech_data = '../data/utterances_feats.pkl'
#speech_data = '../data/cmvn_tensors.pkl'
#speech_data = '../data/skewed_cmvn.pkl'
speech_data = '../../gendata/synth_int_feats.pkl'
labels_data = '../../gendata/synth_int_labels.pkl'
#labels_data = '../data/utterances_labels.pkl'

train_per = 0.6
dev_per = 0.2
print_every = 1000
eval_every = None
VERBOSE = False
STEPTHRU = False

model_name = 'cnn_fc'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_params = {'batch_size': 1,
                     'shuffle': True,
                     'num_workers': 6}

eval_params = {'batch_size': 1,
                          'shuffle': True,
                          'num_workers': 6}
epochs = 1
pad_len = 700
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
        #self.out_channels = 256
        self.out_channels = 128
        self.kernel1 = (9,self.feat_dim)
        #self.kernel2 = (9,1)
        self.kernel2 = (5, 1)
        self.stride1 = (2,self.feat_dim)
        self.stride2 = (2,1)
        self.padding = (2,0)

        self.conv = nn.Sequential(#nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=self.kernel1, stride=self.stride1,
                                  #          padding=self.padding),
                                  #nn.BatchNorm2d(self.hidden_channels),
                                  #nn.Hardtanh(inplace=True),
                                  nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel2, stride=self.stride2,
                                            padding=self.padding),
                                  nn.BatchNorm2d(self.out_channels),
                                  nn.Hardtanh(inplace=True)
                                  )

        # NO RNN VERSION:
        self.maxpool = nn.MaxPool1d(2)
        self.cnn_output_size = math.floor((self.seq_len - self.kernel2[0] + self.padding[0]*2)/self.stride2[0]) + 1
        self.cnn_output_size = int((math.floor(self.cnn_output_size/2)))
        self.cnn_output_size = self.cnn_output_size*self.out_channels


        self.fc1_out = 350

        self.fc1 = nn.Linear(self.cnn_output_size, self.fc1_out)
        self.fc2 = nn.Linear(self.fc1_out,self.num_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.inference_softmax = nn.Softmax(dim=-1)

    def forward(self,x,hidden):
        if VERBOSE: print('Input dims: ', x.shape)
        if VERBOSE: print('Reshaped input dims: ', x.view(1, 1, x.shape[1], 1).shape)
        x = self.conv(x.view(1, 1, x.shape[1], 1))
        if VERBOSE: print('Dims after conv: ',x.shape)
        x = self.maxpool(x.view(x.shape[0],x.shape[1],x.shape[2]))
        if VERBOSE: print('Dims after pooling: ', x.shape)
        x = self.fc1(x.view(x.shape[0],x.shape[1]*x.shape[2]))
        x = self.relu(x)
        if VERBOSE: print('Dims after fc1:', x.shape)
        if STEPTHRU: import pdb;pdb.set_trace()
        x = self.fc2(x)
        #x = self.sigmoid(x)
        if VERBOSE: print('Dims after fc2:', x.shape)
        if STEPTHRU: import pdb;pdb.set_trace()
        if VERBOSE: print('Dims after sigmoid',x.shape)
        #x = self.inference_softmax(x)
        if VERBOSE: print('Dims after softmax:', x.shape)
        if STEPTHRU: import pdb;pdb.set_trace()
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

traingen = data.DataLoader(trainset, **train_params)

print('done')
print('Building model ...')

seq_len = 20

model = SpeechEncoder(seq_len=seq_len,
                      batch_size=train_params['batch_size'],
                      lstm_layers=3,
                      bidirectional=False,
                      #num_classes=2)
                      num_classes=1)

model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
recent_losses = []
timestep = 0

print('done')

print('Baseline eval....')

#logit_evaluate(devset,eval_params,model,device)

print('done')

pred_ones = 0
num_preds = 0
curr_pred_ones = 0
curr_preds = 0

plt_time = []
plt_losses = []
plt_acc = []
plt_dummy = []

print('Training model ...')
for epoch in range(epochs):
    for batch, labels in traingen: # TODO to generalize past binary classification, maybe change labels into one-hot
        batch, labels = batch.to(device),labels.to(device)
        model.zero_grad()
        hidden = model.init_hidden(train_params['batch_size'])
        output,_ = model(batch,hidden)
        if VERBOSE:
            print('output shape: ',output.shape)
            print('labels shape: ',labels.shape)
            print('output: ', output[:,1:].squeeze())
            print('true labels: ',labels.float())
        #print('output: ', output[:,1:].squeeze())

        pred_labels = np.where(np.array(output.cpu().detach()[:,1:].squeeze())>0.5,1,0)
        pred_ones += np.sum(pred_labels)
        num_preds += labels.shape[0]
        curr_pred_ones += pred_ones
        curr_preds += num_preds


        #loss = criterion(output[:, 1:].squeeze(), labels.float())  # No RNN
        loss = criterion(output.view(1), labels.float())  # No RNN
        loss.backward()
        plot_grad_flow(model.named_parameters())
        optimizer.step()
        recent_losses.append(loss.detach())
        if len(recent_losses) > 50:
            recent_losses = recent_losses[1:]

        if timestep % print_every == 1:
            plt_time.append(timestep)
            train_loss = (sum(recent_losses) / len(recent_losses)).item()
            print('Train loss at',timestep,': ', train_loss)
            plt_losses.append(train_loss)
            plt_dummy.append(0)
            process = psutil.Process(os.getpid())
            # print('Memory usage at timestep ', timestep, ':', process.memory_info().rss / 1000000000, 'GB')
            # print('Percent of guesses to this point that are one:',pred_ones/num_preds)
            # print('Percent of guesses since last report that are one:',curr_pred_ones/curr_preds)
            curr_pred_ones = 0
            curr_preds = 0
            # plt.show()
            plt_acc.append(logit_evaluate(devset, eval_params, model, device))

        if eval_every:
            if timestep % eval_every == 1:
                logit_evaluate(devset, eval_params, model, device)
        timestep += 1


print('done')

process = psutil.Process(os.getpid())
print('Memory usage:',process.memory_info().rss/1000000000, 'GB')

plot_results(plt_losses, plt_dummy, plt_acc, plt_time,model_name)
