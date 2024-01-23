"""
Anonymized
"""

from torch import nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import pdb
from model import SpeechEncoder

class SpeechEncoderSeg(SpeechEncoder):

    def __init__(self,
                 seq_len,
                 batch_size,
                 hidden_size=512,
                 bidirectional=True,
                 lstm_layers=3,
                 num_classes=2,
                 dropout=None,
                 include_lstm=True,
                 tok_level_pred=True,
                 postlstm_context=False,
                 cnn_layers=3,
                 feat_dim=16,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 tok_seq_len=None,
                 flatten_method='sum',
                 frame_filter_size=9,
                 frame_pad_size=4,
                 inputs='both',
                 embedding_dim=100,
                 vocab_size=3000,
                 bottleneck_feats=10,
                 use_pretrained=False,
                 weights_matrix=None,
                 return_prefinal=False):
        super(SpeechEncoderSeg,self).__init__(seq_len,
                                           batch_size,
                                           hidden_size=hidden_size,
                                           bidirectional=bidirectional,
                                           lstm_layers=lstm_layers,
                                           num_classes=num_classes,
                                           dropout=dropout,
                                           include_lstm=include_lstm,
                                           tok_level_pred=tok_level_pred,
                                           postlstm_context=postlstm_context,
                                           cnn_layers=cnn_layers,
                                           feat_dim=feat_dim,
                                           device=device,
                                           tok_seq_len=tok_seq_len,
                                           flatten_method=flatten_method,
                                           frame_filter_size=frame_filter_size,
                                           frame_pad_size=frame_pad_size,
                                           inputs=inputs,
                                           embedding_dim=embedding_dim,
                                           vocab_size=vocab_size,
                                           bottleneck_feats=bottleneck_feats,
                                           use_pretrained=use_pretrained,
                                           weights_matrix=weights_matrix,
                                           return_prefinal=return_prefinal)
        self.pause_emb_size = 4
        self.pause_vocab_size = 6
        self.pause_emb = nn.Embedding(self.pause_vocab_size,self.pause_emb_size)

    def forward(self,frames,pause,dur,text,toktimes,hidden):
        '''
        N: number of items in a batch
        C: number of channels
        W: number of frames in signal
        H: number of acoustic features in signal
        '''

        if self.inputs=='both' or self.inputs=='speech':

            toktimes = self.convolve_timestamps(toktimes)

            # in: N x C x W x H

            frames = frames.view(frames.shape[0], 1, frames.shape[1], frames.shape[2])
            frames = self.conv(frames)
            # in: N x C x W x H , where W is compressed and H=1
            pause = pause.view(pause.shape[0],pause.shape[1],1)
            pausedur = torch.cat((pause,dur),dim=-1)
            pausedur = pausedur.permute(1,0,2)
                
        if self.inputs=='both' or self.inputs=='text':

            embeddings = self.emb(text)
            embeddings = embeddings.permute(1,0,2)

        if self.inputs=='both' or self.inputs=='speech':
            speech = frames.squeeze(dim=-1)  # IN: N x C x W x H (where H=1) OUT: N x C x W
            speech = self.token_split(speech, toktimes)
            speech = speech.permute(1,0,2) # Comes out of tokens with dims: batch, seq_len, channels. Need seq_len, batch, channels
            xx = speech
        if self.inputs=='both':
            xx = torch.cat([embeddings,xx,pausedur],dim=2)
        elif self.inputs=='text':
            xx = embeddings

        xx,hidden = self.lstm(xx,hidden) # In: seq_len, batch, channels. Out: seq_len, batch, hidden*2
        xx = self.fc(xx) # In: seq_len, batch, hidden*2. Out: seq_len, batch, num_classes

        return xx, hidden


    def init_hidden(self,batch_size):
        if self.bidirectional:
            h0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(self.device)
            c0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(self.device)
        else:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)

        return (h0,c0)

#class SpeechEncoderPauseDur():
    
