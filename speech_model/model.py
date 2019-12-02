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
from utils import UttDataset,plot_grad_flow,plot_results,UttDatasetWithId,UttDatasetWithToktimes,BurncDatasetSpeech
from evaluate import evaluate,evaluate_lengths,baseline_with_len
import matplotlib.pyplot as plt
import random
import sys
import yaml
import math
import numpy as np

if len(sys.argv) < 2:
    #config = 'conf/swbd-utt.yaml'
    config = 'conf/burnc.yaml'
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
#train_per = float(cfg['train_per'])
#dev_per = float(cfg['dev_per'])

# hyperparameters
tok_level_pred = cfg['tok_level_pred']
include_lstm = cfg['include_lstm']
bidirectional = cfg['bidirectional']
learning_rate = float(cfg['learning_rate'])
hidden_size = int(cfg['hidden_size'])
pad_len = int(cfg['pad_len'])
#datasource = cfg['datasource']
num_epochs = int(cfg['num_epochs'])
LSTM_LAYERS = int(cfg['LSTM_LAYERS'])
dropout = float(cfg['dropout'])
feat_dim = int(cfg['feat_dim'])
context = cfg['context']

# Filenames
#text_data = cfg['text_data']
#speech_data = cfg['speech_data']
#labels_data = cfg['labels_data']
#toktimes_data = cfg['toktimes_data']
all_data = cfg['all_data']
model_name = cfg['model_name']
results_path = cfg['results_path']
results_file = '{}/{}.txt'.format(results_path,model_name)

train_params = cfg['train_params']
eval_params = cfg['eval_params']

seed = cfg['seed']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seeds(seed):
    print(f'setting seed to {seed}')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class SpeechEncoder(nn.Module):

    def __init__(self,
                 seq_len,
                 batch_size,
                 hidden_size=512,
                 bidirectional=True,
                 lstm_layers=3,
                 num_classes=2,
                 dropout=None,
                 include_lstm=True,
                 tok_level_pred=False,
                 feat_dim=16,
                 context=False):
        super(SpeechEncoder,self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.include_lstm = include_lstm
        self.tok_level_pred = tok_level_pred
        self.context = context

        self.feat_dim = feat_dim
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
        if self.include_lstm:

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

        else:
            if not tok_level_pred:
                self.maxpool = nn.MaxPool1d(2)
                self.cnn_output_size = math.floor(
                    (self.seq_len - self.kernel1[0] + self.padding[0] * 2) / self.stride1[0]) + 1
                self.cnn_output_size = math.floor(
                    (self.cnn_output_size - self.kernel2[0] + self.padding[0] * 2) / self.stride2[0]) + 1
                self.cnn_output_size = int(((math.floor(self.cnn_output_size / 2) * self.out_channels)))

                self.fc1_out = 1600

                self.fc1 = nn.Linear(self.cnn_output_size, self.fc1_out)#,dropout=self.dropout)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(self.fc1_out, self.num_classes)

            else:
                self.fc1 = nn.Linear(self.out_channels,int(self.out_channels/2))
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(p=self.dropout)
                self.fc2 = nn.Linear(int(self.out_channels/2),self.num_classes)

    def convolve_timestamps(self,timestamps):
        '''
        Given a tensor of timestamps that correspond to where in the original time signal the token breaks come,
        calculate which frame in the encoder output will correspond to the token break.
        '''
        timestamps = torch.floor((timestamps + (2*self.padding[0]) - self.kernel1[0])/self.stride1[0]) + 1
        timestamps = torch.floor((timestamps + (2*self.padding[0]) - self.kernel2[0])/self.stride2[0]) + 1
        return timestamps

    @staticmethod
    def token_split(input,toktimes):
        toktimes = [int(tim) for tim in toktimes.squeeze().tolist()]
        tokens = []
        for i in range(1,len(toktimes)):
            #import pdb;pdb.set_trace()
            idx1 = toktimes[i-1]
            idx2 = toktimes[i]
            tok = input[:,:,idx1:idx2,:]
            tokens.append(tok)
        return tokens

    @staticmethod
    def token_flatten(toks,context):
        output = []
        for tok in toks:
            summed = tok.sum(dim=2)
            output.append(summed)
            #print(tok.shape)
            #maxed = tok.max(dim=2).values
            #output.append(maxed)

        if context:
            mark_focus = True
            tok_w_context = []
            for i,tok in enumerate(output):
                t_prev = torch.tensor(np.zeros(tok.shape),dtype=torch.float32).to(device) if i==0 else output[i-1]
                t_next = torch.tensor(np.zeros(tok.shape),dtype=torch.float32).to(device) if i==len(output)-1 else output[i+1]
                tok_w_context.append(torch.cat((t_prev,tok,t_next),dim=0))
            out = tok_w_context
        out = torch.cat(output,dim=0)
        #print(out.shape)
        return out

    def forward(self,x,toktimes,hidden):
        '''
        N: number of items in a batch
        C: number of channels
        W: number of frames in signal
        H: number of acoustic features in signal
        '''
        if tok_level_pred:
            toktimes = self.convolve_timestamps(toktimes)
        if VERBOSE: print('Input dims: ', x.view(x.shape[0], 1, x.shape[1], x.shape[2]).shape) # in: N x C x W x H
        x = self.conv(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))
        if VERBOSE: print('Dims after conv: ',x.shape) # in: N x C x W x H , where W is compressed and H=1

        if self.tok_level_pred:

            if self.include_lstm:
                TOKENIZE_FIRST = True # This is the switch to toggle between doing LSTM -> tok vs tok -> LSTM. Not in config file yet.
                if TOKENIZE_FIRST:
                    x = SpeechEncoder.token_split(x, toktimes)  # TODO make work with batches
                    x = SpeechEncoder.token_flatten(x, self.context)  # TODO make work with batches
                    x = x.view(x.shape[0], x.shape[2], x.shape[
                        1])  # Comes out of tokens with dims: seq_len, channels, batch. Need seq_len, batch, channels
                    x,hidden = self.lstm(x,hidden) # In: seq_len, batch, channels. Out: seq_len, batch, hidden*2
                    x = self.fc(x) # In: seq_len, batch, hidden*2. Out: seq_len, batch, num_classes
                    return x,hidden
                else:

                    # NOTE: this path is quite inefficiently written right now. If you continue with this model, rewrite.
                    # (The main culprit is the two reshapings to make it cooperate with the LSTM which isn't batch-first
                    # that is easy enough to change if this is going to be used more heavily, but didn't want to change the
                    # constructor function for the sake of this version of the model, if it's not gonna be used a lot)

                    x = x.view(x.shape[0], x.shape[1], x.shape[2]).transpose(1, 2).transpose(0,1).contiguous()  # here: W x N x C
                    x,hidden = self.lstm(x,hidden) # In: seq_len, batch, channels. Out: seq_len, batch, hidden*2
                    x = x.transpose(0,1).transpose(1,2).contiguous()
                    x = x.view(x.shape[0],x.shape[1],x.shape[2],1)
                    x = SpeechEncoder.token_split(x, toktimes)  # TODO make work with batches
                    x = SpeechEncoder.token_flatten(x, self.context)  # TODO make work with batches
                    x = x.view(x.shape[0],x.shape[2],x.shape[1])
                    x = self.fc(x) # In: seq_len, batch, hidden*2. Out: seq_len, batch, num_classes

                    return x,hidden


            else:
                x = SpeechEncoder.token_split(x, toktimes)  # TODO make work with batches
                x = SpeechEncoder.token_flatten(x, self.context)  # TODO make work with batches
                x = x.view(x.shape[0], x.shape[2], x.shape[1])  # Comes out of tokens with dims: seq_len, channels, batch. Need seq_len, batch, channels
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x,hidden

        else:
            if self.include_lstm:
                x = x.view(x.shape[0], x.shape[1], x.shape[2]).transpose(1, 2).transpose(0, 1).contiguous() # here: W x N x C
                if VERBOSE: print('Dims going into lstm: ', x.shape)
                x, hidden = self.lstm(x.view(x.shape[0], x.shape[1], x.shape[2]), hidden)
                if VERBOSE: print('Dims after lstm:', x.shape) # here: W x N x lstm_hidden_size
                x = x[-1,:,:] # TAKE LAST TIMESTEP # here: 1 x N x lstm_hidden_size
                #x = torch.mean(x,0)
                if VERBOSE: print('Dims after compression:', x.shape)
                x = self.fc(x.view(1, x.shape[0], self.lin_input_size))  # in 1 x N? x lstm_hidden_size
                if VERBOSE: print('Dims after fc:', x.shape)
                return x,hidden
            else:
                x = self.maxpool(x.view(x.shape[0], x.shape[1], x.shape[2]))
                if VERBOSE: print('Dims after pooling: ', x.shape)
                x = self.fc1(x.view(x.shape[0], x.shape[1] * x.shape[2]))
                x = self.relu(x)
                if VERBOSE: print('Dims after fc1:', x.shape)
                x = self.fc2(x)
                if VERBOSE: print('Dims after fc2:', x.shape)
                return x, hidden

    def init_hidden(self,batch_size):
        if self.bidirectional:
            h0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(device)
            c0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(device)
        else:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(device)

        return (h0,c0)

"""
print('Loading data ...')
with open(labels_data,'rb') as f:
    labels_dict = pickle.load(f)
with open(speech_data,'rb') as f:
    feat_dict = pickle.load(f)
with open(toktimes_data,'rb') as f:
    toktimes_dict = pickle.load(f)

set_seeds(seed)

#all_ids = list(labels_dict.keys())
all_ids = list(feat_dict.keys())
random.shuffle(all_ids)


#train_ids = all_ids[:int(len(all_ids)*train_per)]
#dev_ids = all_ids[int(len(all_ids)*train_per):int(len(all_ids)*(train_per+dev_per))]
#test_ids = all_ids[int(len(all_ids)*(train_per+dev_per)):]

with open(cfg['datasplit'],'r') as f:
    all_ids = yaml.load(f,yaml.FullLoader)

train_ids = all_ids['train']
dev_ids = all_ids['dev']
test_ids = all_ids['test']


if tok_level_pred:
    trainset = UttDatasetWithToktimes(train_ids,feat_dict,labels_dict,toktimes_dict,pad_len)
else:
    trainset = UttDataset(train_ids,feat_dict,labels_dict,pad_len)

if tok_level_pred:
    devset = UttDatasetWithToktimes(dev_ids,feat_dict,labels_dict,toktimes_dict,pad_len)
elif LENGTH_ANALYSIS:
    devset = UttDatasetWithId(dev_ids, feat_dict, labels_dict, pad_len)
else:
    devset = UttDataset(dev_ids,feat_dict,labels_dict,pad_len)


old_trainset = UttDatasetWithToktimes(train_ids,feat_dict,labels_dict,toktimes_dict,pad_len)
old_devset = UttDatasetWithToktimes(dev_ids, feat_dict, labels_dict, toktimes_dict, pad_len)

set_seeds(seed)
old_traingen = data.DataLoader(old_trainset, **train_params)

#######################
#for key in feat_dict:
#    assert(feat_dict[key].shape[1]==16)
#######################

"""
with open(all_data,'rb') as f:
    data_dict = pickle.load(f)

trainset = BurncDatasetSpeech(cfg, data_dict, pad_len, mode='train')
devset = BurncDatasetSpeech(cfg, data_dict, pad_len, mode='dev')

set_seeds(seed)
traingen = data.DataLoader(trainset, **train_params)

print('done')
print('Building model ...')

set_seeds(seed)

"""
#################################

for (id1,(batch1,toktimes1),labels1),(id2, (batch2,toktimes2),labels2) in zip(old_traingen,traingen):
    print(id1)
    print(id2)
    print('-------------')
    print(batch1)
    print(batch2)
    print('-------------')
    print(toktimes1)
    print(toktimes2)
    print('-------------')
    print(labels1)
    print(labels2)
    print("==================")
    import pdb;pdb.set_trace()


#################################
"""

model = SpeechEncoder(seq_len=pad_len,
                      batch_size=train_params['batch_size'],
                      lstm_layers=LSTM_LAYERS,
                      bidirectional=False,
                      num_classes=1,
                      dropout=dropout,
                      include_lstm=include_lstm,
                      tok_level_pred=tok_level_pred,
                      feat_dim=feat_dim,
                      context=context)

model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
recent_losses = []
timestep = 0

print('done')


plt_time = []
plt_losses = []
plt_acc = []
plt_train_acc = []


print('Baseline eval....')

if LENGTH_ANALYSIS:
    baseline_with_len(devset, eval_params)
    plt_acc.append(evaluate_lengths(devset, eval_params, model, device))
    plt_train_acc.append(evaluate_lengths(trainset,eval_params,model,device))
else:
    plt_acc.append(evaluate(devset, eval_params, model, device,tok_level_pred=tok_level_pred))
    plt_train_acc.append(evaluate(trainset, eval_params, model, device,tok_level_pred=tok_level_pred))
plt_losses.append(0)
plt_time.append(0)

print('done')

set_seeds(seed)



print('Training model ...')
max_grad = float("-inf")
min_grad = float("inf")
for epoch in range(num_epochs):
    for id, (batch, toktimes), labels in traingen:

        model.train()
        batch,labels = batch.to(device),labels.to(device)
        #if not batch.shape[0] < train_params['batch_size']:
         #print("+++++++++++++++++")
        #print(f'timestep {timestep}')

        model.zero_grad()
        #hidden = model.init_hidden(train_params['batch_size'])
        curr_bat_size = batch.shape[0]
        hidden = model.init_hidden(curr_bat_size)
        output,_ = model(batch,toktimes,hidden)

        ##############

        #para = data_dict['utt2para'][id[0]]
        #utts = data_dict['para2utt'][para]
        #print(id[0])
        #print(utts)
        #print(id[0]==utts[-1])
        #import pdb;pdb.set_trace()

        ##############

        if VERBOSE:
            print('output shape: ',output.shape)
            print('labels shape: ',labels.shape)
            print('output: ',output.cpu().detach().squeeze())
            print('true labels: ',labels.float())

        """
        print(id)
        print('in')
        print(batch)
        print('out')
        print(output)
        print('==============================================')
        """
        if tok_level_pred:
            loss = criterion(output.view(output.shape[1],output.shape[0]), labels.float())
        else:
            #loss = criterion(output.view(train_params['batch_size']), labels.float())
            loss = criterion(output.view(curr_bat_size), labels.float())
        loss.backward()
        plot_grad_flow(model.named_parameters())


        """
        # Check for exploding gradients
        for n, p in model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                curr_min, curr_max = p.grad.min(), p.grad.max()
                if curr_min <= min_grad:
                    min_grad = curr_min
                if curr_max >= max_grad:
                    max_grad = curr_max
        print(f'min/max grad: {min_grad}, {max_grad}')
        """
        #import pdb;pdb.set_trace()


        if torch.sum(torch.isnan(batch)).item() > 0:
            import pdb;pdb.set_trace()


        torch.nn.utils.clip_grad_norm(model.parameters(), 5)

        optimizer.step()
        recent_losses.append(loss.detach())

        if len(recent_losses) > 50:
            recent_losses = recent_losses[1:]

        if timestep % print_every == 1:
            plt_time.append(timestep)
            train_loss = (sum(recent_losses)/len(recent_losses)).item()
            print('Train loss at',timestep,': ',train_loss)
            process = psutil.Process(os.getpid())

            plt_losses.append(train_loss)
            #plt.show()
            if LENGTH_ANALYSIS:
                print('train')
                plt_train_acc.append(evaluate_lengths(trainset, eval_params, model, device))
                print('dev')
                plt_acc.append(evaluate_lengths(devset, eval_params, model, device))
            else:
                print('train')
                plt_train_acc.append(evaluate(trainset, eval_params, model, device,tok_level_pred=tok_level_pred))
                print('dev')
                plt_acc.append(evaluate(devset, eval_params, model, device,tok_level_pred=tok_level_pred))
        if eval_every:
            if timestep % eval_every == 1 and not timestep==1:
                evaluate(devset,eval_params,model,device,tok_level_pred=tok_level_pred)
        timestep += 1


print('done')

process = psutil.Process(os.getpid())
print('Memory usage:',process.memory_info().rss/1000000000, 'GB')

plot_results(plt_losses, plt_train_acc, plt_acc, plt_time,model_name,results_path)

def print_progress(progress, info='', bar_len=20):
	filled = int(progress*bar_len)
	print('\r[{}{}] {:.2f}% {}'.format('=' * filled, ' ' * (bar_len-filled), progress*100, info), end='')