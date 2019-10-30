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
from utils import UttDataset,plot_grad_flow,plot_results,UttDatasetWithId
from evaluate import evaluate,logit_evaluate,logit_evaluate_lengths,baseline_with_len
import matplotlib.pyplot as plt
import random
import sys
import yaml
random.seed(123)

if len(sys.argv) < 2:
    config = 'conf/swbd-utt.yaml'
else:
    config = sys.argv[1]

with open(config,'r') as f:
    cfg = yaml.load(f,yaml.FullLoader)


VERBOSE = cfg['VERBOSE']
LENGTH_ANALYSIS = cfg['LENGTH_ANALYSIS']
print_every = int(cfg['print_every'])
eval_every = cfg['eval_every']
if eval_every:
    eval_every = int(eval_every)
train_per = float(cfg['train_per'])
dev_per = float(cfg['dev_per'])

# hyperparameters
bidirectional = cfg['bidirectional']
learning_rate = float(cfg['learning_rate'])
hidden_size = int(cfg['hidden_size'])
pad_len = int(cfg['pad_len'])
datasource = cfg['datasource']
num_epochs = int(cfg['num_epochs'])
LSTM_LAYERS = int(cfg['LSTM_LAYERS'])
dropout = float(cfg['dropout'])

# Filenames
text_data = cfg['text_data']
speech_data = cfg['speech_data']
labels_data = cfg['labels_data']
model_name = cfg['model_name']
results_path = cfg['results_path']
results_file = '{}/{}.txt'.format(results_path,model_name)

train_params = cfg['train_params']
eval_params = cfg['eval_params']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpeechEncoder(nn.Module):
    def __init__(self,
                 seq_len,
                 batch_size,
                 hidden_size=512,
                 bidirectional=True,
                 lstm_layers=3,
                 num_classes=2,
                 dropout=None):
        super(SpeechEncoder,self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        self.dropout = dropout

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
                                  #nn.ReLU(inplace=True),
                                  nn.Hardtanh(inplace=True),
                                  nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=self.kernel2, stride=self.stride2,
                                            padding=self.padding),
                                  nn.BatchNorm2d(self.out_channels),
                                  #nn.ReLU(inplace=True)
                                  nn.Hardtanh(inplace=True)
                                  )


        # RNN VERSION:
        rnn_input_size = self.out_channels # This is what I think the input size of the LSTM should be -- channels, not time dim
        self.lstm = nn.LSTM(input_size=rnn_input_size,
                            #batch_first=True,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=self.bidirectional,
                            dropout=self.dropout)

        if self.bidirectional:
            self.lin_input_size = self.hidden_size * 2
        else:
            self.lin_input_size = self.hidden_size

        self.fc = nn.Linear(self.lin_input_size, self.num_classes, bias=False)

    def forward(self,x,hidden):
        if VERBOSE: ('Input dims: ', x.view(x.shape[0], 1, x.shape[1], x.shape[2]).shape)
        x = self.conv(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))
        if VERBOSE: print('Dims after conv: ',x.shape)
        x = x.view(x.shape[0],x.shape[1],x.shape[2]).transpose(1,2).transpose(0,1).contiguous()
        if VERBOSE: print('Dims going into lstm: ',x.shape)
        x,hidden = self.lstm(x.view(x.shape[0],x.shape[1],x.shape[2]),hidden)
        if VERBOSE: print('Dims after lstm:', x.shape)
        x = x[-1,:,:] # TAKE LAST TIMESTEP
        #x = torch.mean(x,0)
        if VERBOSE: print('Dims after compression:', x.shape)
        x = self.fc(x.view(1, x.shape[0], self.lin_input_size))
        if VERBOSE: print('Dims after fc:', x.shape)
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


trainset = UttDataset(train_ids,feat_dict,labels_dict,pad_len)
if LENGTH_ANALYSIS:
    devset = UttDatasetWithId(dev_ids, feat_dict, labels_dict, pad_len)
else:
    devset = UttDataset(dev_ids,feat_dict,labels_dict,pad_len)



traingen = data.DataLoader(trainset, **train_params)

print('done')
print('Building model ...')

model = SpeechEncoder(seq_len=pad_len,
                      batch_size=train_params['batch_size'],
                      lstm_layers=LSTM_LAYERS,
                      bidirectional=False,
                      num_classes=1,
                      dropout=dropout)

model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
recent_losses = []
timestep = 0

print('done')

print('Baseline eval....')

if LENGTH_ANALYSIS:
    baseline_with_len(devset, eval_params)
    logit_evaluate_lengths(devset, eval_params, model, device)
else:
    logit_evaluate(devset, eval_params, model, device)

print('done')

plt_time = []
plt_losses = []
plt_acc = []
plt_train_acc = []


print('Training model ...')
for epoch in range(num_epochs):
    for batch,labels in traingen:
        model.train()
        batch,labels = batch.to(device),labels.to(device)
        if not batch.shape[0] < train_params['batch_size']:
            model.zero_grad()
            hidden = model.init_hidden(train_params['batch_size'])
            output,_ = model(batch,hidden)
            if VERBOSE:
                print('output shape: ',output.shape)
                print('labels shape: ',labels.shape)
                print('output: ',output.cpu().detach().squeeze())
                print('true labels: ',labels.float())
            #print('output: ', output.cpu().detach().squeeze())
            #import pdb;pdb.set_trace()

            loss = criterion(output.view(train_params['batch_size']), labels.float())  # With RNN
            loss.backward()
            plot_grad_flow(model.named_parameters())
            optimizer.step()
            recent_losses.append(loss.detach())

            if len(recent_losses) > 50:
                recent_losses = recent_losses[1:]

            if timestep % print_every == 1:
                plt_time.append(timestep)
                train_loss = (sum(recent_losses)/len(recent_losses)).item()
                print('Train loss at',timestep,': ',train_loss)
                process = psutil.Process(os.getpid())
                #print('Memory usage at timestep ', timestep, ':', process.memory_info().rss / 1000000000, 'GB')

                plt_losses.append(train_loss)
                #plt.show()
                if LENGTH_ANALYSIS:
                    print('train')
                    plt_train_acc.append(logit_evaluate_lengths(trainset, eval_params, model, device))
                    print('dev')
                    plt_acc.append(logit_evaluate_lengths(devset, eval_params, model, device))
                else:
                    print('train')
                    plt_train_acc.append(logit_evaluate(trainset, eval_params, model, device))
                    print('dev')
                    plt_acc.append(logit_evaluate(devset, eval_params, model, device))
            if eval_every:
                if timestep % eval_every == 1 and not timestep==1:
                    logit_evaluate(devset,eval_params,model,device)
            timestep += 1
            #import pdb;pdb.set_trace()
        else:
            print('Batch of size',batch.shape,'rejected')
            print('Last batch:')
            print(batch)

print('done')

process = psutil.Process(os.getpid())
print('Memory usage:',process.memory_info().rss/1000000000, 'GB')

plot_results(plt_losses, plt_train_acc, plt_acc, plt_time,model_name)

