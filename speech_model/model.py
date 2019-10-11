"""
Based on https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
by Sean Naren

Directly copied functions noted.

Modified by: Elizabeth Nielsen
"""

import pandas as pd
import pickle
from torch import nn
import torch
import psutil
import os
import time

text_data = '../data/utterances.txt'
speech_data = '../data/utterances_feats.pkl'


df = pd.read_csv(text_data,sep='\t')
utt_keys = set(df.iloc[:,1].tolist())


t1 = time.time()
print("Loading in dict ...")
with open(speech_data,'rb') as f:
    utt_dict = pickle.load(f)
t2 = time.time()
print("Time to load: ",t2-t1)

process = psutil.Process(os.getpid())
print('Memory usage:',process.memory_info().rss/1000000000, 'GB')

class MaskConv(nn.Module):
    """
    Copied directly from Sean Naren
    """
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths

class SequenceWise(nn.Module):
    """
    Copied directly from Sean Naren
    """
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class SpeechEncoder(nn.Module):
    def __init__(self,
                 rnn_type=nn.LSTM,
                 rnn_hidden_size=512,
                 bidirectional=True,
                 num_rnn_layers=3,
                 seq_len=800,
                 num_classes=2):
        super(SpeechEncoder,self).__init__()
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional
        self.num_rnn_layers = num_rnn_layers
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
        self.seq_len = seq_len

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=self.kernel1, stride=self.stride1,
                      padding=self.padding),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True), # TODO figure out if inplace is correct
            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=self.kernel2, stride=self.stride2,
                      padding=self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        ))

        rnn_input_size = self.seq_len
        rnn_input_size = (rnn_input_size- self.kernel1[0] + 2 * self.padding[0]) / (1 + self.stride1[0])
        rnn_input_size = (rnn_input_size - self.kernel2[0] + 2 * self.padding[0]) / (1 + self.stride2[0])
        self.lstm = nn.LSTM(input_size=rnn_input_size,
                            hidden_size=self.rnn_hidden_size,
                            num_layers=self.num_rnn_layers)


        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, self.num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = nn.Softmax()

