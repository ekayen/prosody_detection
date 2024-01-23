"""
Anonymized
"""

from torch import nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import pdb

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
        super(SpeechEncoder,self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.include_lstm = include_lstm
        self.postlstm_context = postlstm_context
        self.cnn_layers = cnn_layers

        self.tok_level_pred = tok_level_pred
        self.device = device
        self.flatten_method = flatten_method
        self.tok_seq_len = tok_seq_len

        self.frame_filter_size = frame_filter_size
        self.frame_pad_size = frame_pad_size
        self.feat_dim = feat_dim
        self.in_channels = 1
        self.hidden_channels = 128
        self.out_channels = 256
        self.kernel1 = (self.frame_filter_size,self.feat_dim)
        self.kernel2 = (self.frame_filter_size,1)
        self.stride1 = (2,self.feat_dim)
        self.stride2 = (2,1)
        self.padding = (self.frame_pad_size,0)

        self.inputs = inputs
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.bottleneck_feats = bottleneck_feats
        self.use_pretrained = use_pretrained
        self.weights_matrix = weights_matrix

        seg = False
        if seg:
            self.pause_size = 1
            self.dur_size = 2
        else:
            self.pause_size = 0
            self.dur_size = 0
                                                                                                                                                               
        if inputs=='text' or inputs=='both':

            self.emb = nn.Embedding(vocab_size+2,embedding_dim)
            if self.use_pretrained:
                self.emb.load_state_dict({'weight': self.weights_matrix})
                self.emb.weight.requires_grad = False

        if inputs=='speech' or inputs=='both':
            self.conv = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=self.kernel1, stride=self.stride1,
                                                padding=self.padding),
                                      nn.BatchNorm2d(self.hidden_channels),
                                      nn.Hardtanh(inplace=True),
                                      nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=self.kernel2, stride=self.stride2,
                                                padding=self.padding),
                                      nn.BatchNorm2d(self.out_channels),
                                      nn.Hardtanh(inplace=True),
                                      nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel2,
                                                stride=self.stride2,
                                                padding=self.padding),
                                      nn.BatchNorm2d(self.out_channels),
                                      nn.Hardtanh(inplace=True) )


        if self.inputs=='speech':
            rnn_input_size = self.out_channels # This is what I think the input size of the LSTM should be -- channels, not time dim
        elif self.inputs=='text':
            rnn_input_size = self.embedding_dim
        elif self.inputs=='both':
            rnn_input_size = self.out_channels + self.embedding_dim

        # Extra feats for segmentation task:
        if seg and self.inputs == 'speech' or self.inputs == 'both':
            rnn_input_size += self.pause_size + self.dur_size

        if self.include_lstm:
            ## CNN+LSTM version
            if self.tok_level_pred:
                print(f'rnn_input_size: {rnn_input_size}')
                self.lstm = nn.LSTM(input_size=rnn_input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.lstm_layers,
                                bidirectional=self.bidirectional,
                                dropout=self.dropout)

                if self.bidirectional:
                    self.fc1_in = self.hidden_size * 2
                else:
                    self.fc1_in = self.hidden_size

                self.fc = nn.Linear(self.fc1_in, self.num_classes, bias=False)
            else:
                raise NotImplementedError
            
        else:
            ## CNN only version
            self.fc1_in = rnn_input_size
            self.fc1_out = self.bottleneck_feats

            if not self.tok_level_pred:
                
                ##############################################
                ## REPLICATION OF STEHWIEN ET AL.          ###
                ##############################################
                # CNN only, 3 tok window, one pred per 3 toks
                self.maxpool = nn.MaxPool1d(2)
                # layer 1
                self.cnn_output_size = math.floor(
                    (self.seq_len - self.kernel1[0] + self.padding[0] * 2) / self.stride1[0]) + 1
                # layer 2
                self.cnn_output_size = math.floor(
                    (self.cnn_output_size - self.kernel2[0] + self.padding[0] * 2) / self.stride2[0]) + 1
                # layer 3
                if self.cnn_layers>=3:
                    self.cnn_output_size = math.floor(
                        (self.cnn_output_size - self.kernel2[0] + self.padding[0] * 2) / self.stride2[0]) + 1
                if self.cnn_layers>=4:
                    self.cnn_output_size = math.floor(
                        (self.cnn_output_size - self.kernel2[0] + self.padding[0] * 2) / self.stride2[0]) + 1
                self.cnn_output_size = int(((math.floor(self.cnn_output_size / 2) * self.out_channels)))
                print("self.cnn_output_size")
                print(self.cnn_output_size)
                if self .inputs=='speech':
                    self.fc1_in = self.cnn_output_size
                    #self.fc1_in = self.out_channels
                elif self.inputs=='both':
                    self.fc1_in = self.cnn_output_size + (self.embedding_dim * self.tok_seq_len)
                    #self.fc1_in = self.out_channels + (self.embedding_dim * self.tok_seq_len)
                elif self.inputs=='text':
                    self.fc1_in = self.embedding_dim * self.tok_seq_len


                self.fc1 = nn.Linear(self.fc1_in, self.fc1_out)#,dropout=self.dropout)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(self.fc1_out, self.num_classes)

            else:
                #########################################
                # CNN-only version of my model        ###
                #########################################
                # CNN only, whole utterance, one prediction per tok
                
                self.fc1 = nn.Linear(self.fc1_in,self.fc1_out) # TODO make the same as above
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(p=self.dropout)
                self.fc2 = nn.Linear(self.fc1_out,self.num_classes)

            
 
    def convolve_timestamps(self,timestamps):
        '''
        Given a tensor of timestamps that correspond to where in the original time signal the token breaks come,
        calculate which frame in the encoder output will correspond to the token break.
        '''
        # EKN 2021 this is for 3 CNN layers
        timestamps = torch.floor((timestamps + (2*self.padding[0]) - self.kernel1[0])/self.stride1[0]) + 1
        timestamps = torch.floor((timestamps + (2*self.padding[0]) - self.kernel2[0])/self.stride2[0]) + 1
        timestamps = torch.floor((timestamps + (2*self.padding[0]) - self.kernel2[0])/self.stride2[0]) + 1

        return timestamps

    def token_split(self,inp,toktimes):
        batch_size = inp.shape[0]
        instances = []
        for jj in range(batch_size):
            tokens = []
            instance = inp[jj:jj+1,:,:]
            curr_toktimes = toktimes[jj:jj+1,:].squeeze()
            curr_toktimes = np.trim_zeros(np.array(curr_toktimes,dtype=np.int64),trim='b').tolist()
            for i in range(1,len(curr_toktimes)):
                idx1 = curr_toktimes[i-1]
                idx2 = curr_toktimes[i]
                tok = instance[:,:,idx1:idx2]
                tokens.append(tok)
            tokens = self.token_flatten(tokens).unsqueeze(dim=0)
            if tokens.shape[1] < self.tok_seq_len:
                dff = self.tok_seq_len - tokens.shape[1]
                tokens = F.pad(tokens,pad=(0,0,0,dff,0,0),mode='constant')
            else:
                tokens = tokens[:,:self.tok_seq_len,:]
            instances.append(tokens)

        out = torch.cat(instances,dim=0)
        return out

    def token_flatten(self,toks):
        output = []
        for tok in toks:
            if self.flatten_method=='sum':
                summed = tok.sum(dim=2)
            elif self.flatten_method=='max':
                if tok.shape[-1]==0:
                    summed = torch.zeros(tok.shape[0],tok.shape[1]).to(self.device)
                else:
                    summed,_ = tok.max(dim=2)
            output.append(summed)
        out = torch.cat(output,dim=0)
        return out

    def forward(self,
                speech=None,
                text=None,
                toktimes=None,
                hidden=None):
        '''
        N: number of items in a batch
        C: number of channels
        W: number of frames in signal
        H: number of acoustic features in signal
        '''

        # PROCESS INPUT THROUGH CNN for all model variants
        # Small changes marked:
        if self.inputs=='both' or self.inputs=='speech':
            assert not speech == None
            assert not toktimes == None
            toktimes = self.convolve_timestamps(toktimes)

            # in: N x C x W x H
            speech = self.conv(speech.view(speech.shape[0], 1, speech.shape[1], speech.shape[2]))
            # in: N x C x W x H , where W is compressed and H=1
            post_cnn_feats = speech
            speech = speech.squeeze(dim=-1)  # IN: N x C x W x H (where H=1) OUT: N x C x W
            if self.tok_level_pred: # only do token splitting for full utterance mode
                speech = self.token_split(speech, toktimes)
                speech = speech.permute(1,0,2) # Comes out of tokens with dims: batch, seq_len, channels. Need seq_len, batch, channels
            if self.inputs == 'speech':
                x = speech
            
        if self.inputs=='both' or self.inputs=='text':
            assert not text == None
            embeddings = self.emb(text)
            if self.tok_level_pred or self.include_lstm:
                embeddings = embeddings.permute(1,0,2) # 2024 this step not needed for 3 tok, cnn only. Check if that works
            if self.inputs == 'text':
                x = embeddings
        
        if self.inputs=='both':
            x = torch.cat([embeddings,speech],dim=2)

        # AT THIS POINT, models diverge
        
        if self.include_lstm:
            if self.tok_level_pred:
                # My model, CNN+LSTM, whole utterances, one pred per tok:
                x,hidden = self.lstm(x,hidden) # In: seq_len, batch, channels. Out: seq_len, batch, hidden*2
                x = self.fc(x) # In: seq_len, batch, hidden*2. Out: seq_len, batch, num_classes
                return x, hidden
            else:
                # My model, but with 3 tok -> 1 pred
                raise NotImplementedError
            

        else:
            if self.tok_level_pred:
                # CNN only, whole utterances, one pred per tok
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x,hidden

            else:
                # CNN only, 3 tok, one pred total (Replication of Stehwien et al.)
                if self.inputs=='speech' or self.inputs == 'both':
                    x = self.maxpool(speech.view(speech.shape[0], speech.shape[1], speech.shape[2]))
                if self.inputs=='both':
                    x = torch.cat([speech,embeddings],dim=2) # TODO test
                elif self.inputs=='text':
                    x = embeddings

                x = self.fc1(x.view(x.shape[0], x.shape[1] * x.shape[2]))
                x = self.relu(x)
                x = self.fc2(x)
        
                return x, hidden
   

        #return x, hidden


    def init_hidden(self,batch_size):
        if self.bidirectional:
            h0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(self.device)
            c0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(self.device)
        else:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)

        return (h0,c0)

#class SpeechEncoderPauseDur():
    
