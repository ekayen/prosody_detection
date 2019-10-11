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
                 rnn_hidden_size=512,
                 bidirectional=True,
                 num_rnn_layers=3,
                 num_classes=2):
        super(SpeechEncoder,self).__init__()
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

        rnn_input_size = self.seq_len # TODO FIRST figure out what the seq_len is, in fact
        rnn_input_size = (rnn_input_size - self.kernel1[0] + 2 * self.padding[0]) / (1 + self.stride1[0])
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


    def forward(self, x, lengths):
        lengths = lengths.int() # TODO figure out if .cpu() is needed here
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()


t1 = time.time()
print("Loading in dict ...")
with open(speech_data,'rb') as f:
    utt_dict = pickle.load(f)
t2 = time.time()
print("Time to load: ",t2-t1)

model = SpeechEncoder(rnn_hidden_size=512,bidirectional=True,num_rnn_layers=3,num_classes=2)

instance = utt_dict[utt_keys[57]]

model(instance,instance.shape[0])


process = psutil.Process(os.getpid())
print('Memory usage:',process.memory_info().rss/1000000000, 'GB')
