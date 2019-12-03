from torch import nn
import torch
import math
import numpy as np

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
                 context=False,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
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
        self.device = device

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
            if not self.tok_level_pred:
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


    def token_split(self,input,toktimes):
        toktimes = [int(tim) for tim in toktimes.squeeze().tolist()]
        tokens = []
        for i in range(1,len(toktimes)):
            #import pdb;pdb.set_trace()
            idx1 = toktimes[i-1]
            idx2 = toktimes[i]
            tok = input[:,:,idx1:idx2,:]
            tokens.append(tok)
        tokens = self.token_flatten(tokens)
        return tokens


    def token_flatten(self,toks):
        output = []
        for tok in toks:
            summed = tok.sum(dim=2)
            output.append(summed)
            #print(tok.shape)
            #maxed = tok.max(dim=2).values
            #output.append(maxed)

        if self.context:
            mark_focus = True
            tok_w_context = []
            for i,tok in enumerate(output):
                t_prev = torch.tensor(np.zeros(tok.shape),dtype=torch.float32).to(self.device) if i==0 else output[i-1]
                t_next = torch.tensor(np.zeros(tok.shape),dtype=torch.float32).to(self.device) if i==len(output)-1 else output[i+1]
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
        if self.tok_level_pred:
            toktimes = self.convolve_timestamps(toktimes)
        # in: N x C x W x H
        x = self.conv(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))
        # in: N x C x W x H , where W is compressed and H=1

        if self.tok_level_pred:

            if self.include_lstm:
                TOKENIZE_FIRST = True # This is the switch to toggle between doing LSTM -> tok vs tok -> LSTM. Not in config file yet.
                if TOKENIZE_FIRST:
                    x = self.token_split(x, toktimes)  # TODO make work with batches
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
                    x = self.token_split(x, toktimes)  # TODO make work with batches
                    x = x.view(x.shape[0],x.shape[2],x.shape[1])
                    x = self.fc(x) # In: seq_len, batch, hidden*2. Out: seq_len, batch, num_classes

                    return x,hidden


            else:
                x = self.token_split(x, toktimes)  # TODO make work with batches
                x = x.view(x.shape[0], x.shape[2], x.shape[1])  # Comes out of tokens with dims: seq_len, channels, batch. Need seq_len, batch, channels
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x,hidden

        else:
            if self.include_lstm:
                x = x.view(x.shape[0], x.shape[1], x.shape[2]).transpose(1, 2).transpose(0, 1).contiguous() # here: W x N x C

                x, hidden = self.lstm(x.view(x.shape[0], x.shape[1], x.shape[2]), hidden)
                # here: W x N x lstm_hidden_size
                x = x[-1,:,:] # TAKE LAST TIMESTEP # here: 1 x N x lstm_hidden_size
                #x = torch.mean(x,0)
                x = self.fc(x.view(1, x.shape[0], self.lin_input_size))  # in 1 x N? x lstm_hidden_size
                return x,hidden
            else:
                x = self.maxpool(x.view(x.shape[0], x.shape[1], x.shape[2]))
                x = self.fc1(x.view(x.shape[0], x.shape[1] * x.shape[2]))
                x = self.relu(x)
                x = self.fc2(x)
                return x, hidden

    def init_hidden(self,batch_size):
        if self.bidirectional:
            h0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(self.device)
            c0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(self.device)
        else:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)

        return (h0,c0)
