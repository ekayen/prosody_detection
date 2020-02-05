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
                 tok_level_pred=False,
                 feat_dim=16,
                 postlstm_context=False,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 tok_seq_len=None,
                 flatten_method='sum',
                 frame_filter_size=9,
                 frame_pad_size=4,
                 cnn_layers=2,
                 inputs='speech',
                 embedding_dim=100,
                 vocab_size=3000,
                 bottleneck_feats=10,
                 use_pretrained=False,
                 weights_matrix=None):
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
        self.postlstm_context = postlstm_context
        self.device = device
        self.flatten_method = flatten_method
        self.cnn_layers = cnn_layers

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

        if inputs=='text' or inputs=='both':
            self.emb = nn.Embedding(vocab_size+2,embedding_dim)
            if self.use_pretrained:
                self.emb.load_state_dict({'weight': self.weights_matrix})
                self.emb.weight.requires_grad = False



        if inputs=='speech' or inputs=='both':
            if self.cnn_layers==2:
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
            elif self.cnn_layers==3:
                self.conv = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=self.kernel1, stride=self.stride1,
                                                    padding=self.padding),
                                          nn.BatchNorm2d(self.hidden_channels),
                                          #nn.ReLU(inplace=True),
                                          nn.Hardtanh(inplace=True),

                                          nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=self.kernel2, stride=self.stride2,
                                                    padding=self.padding),
                                          nn.BatchNorm2d(self.out_channels),
                                          #nn.ReLU(inplace=True)
                                          nn.Hardtanh(inplace=True),

                                          nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel2,
                                                    stride=self.stride2,
                                                    padding=self.padding),
                                          nn.BatchNorm2d(self.out_channels),
                                          # nn.ReLU(inplace=True)
                                          nn.Hardtanh(inplace=True)
                                          )
            elif self.cnn_layers==4:
                self.conv = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=self.kernel1, stride=self.stride1,
                                                    padding=self.padding),
                                          nn.BatchNorm2d(self.hidden_channels),
                                          #nn.ReLU(inplace=True),
                                          nn.Hardtanh(inplace=True),

                                          nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=self.kernel2, stride=self.stride2,
                                                    padding=self.padding),
                                          nn.BatchNorm2d(self.out_channels),
                                          #nn.ReLU(inplace=True)
                                          nn.Hardtanh(inplace=True),

                                          nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel2,
                                                    stride=self.stride2,
                                                    padding=self.padding),
                                          nn.BatchNorm2d(self.out_channels),
                                          # nn.ReLU(inplace=True)
                                          nn.Hardtanh(inplace=True),

                                          nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel2,
                                                    stride=self.stride2,
                                                    padding=self.padding),
                                          nn.BatchNorm2d(self.out_channels),
                                          # nn.ReLU(inplace=True)
                                          nn.Hardtanh(inplace=True)
                                          )

        # RNN VERSION:
        if self.include_lstm:

            if self.inputs=='speech':
                rnn_input_size = self.out_channels # This is what I think the input size of the LSTM should be -- channels, not time dim
            elif self.inputs=='text':
                rnn_input_size = self.embedding_dim
            elif self.inputs=='both':
                rnn_input_size = self.out_channels + self.embedding_dim
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
                ##############################################
                ## REPLICATION OF STEHWIEN ET AL.          ###
                ##############################################

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

                if self.inputs=='speech':
                    self.fc1_in = self.cnn_output_size
                elif self.inputs=='both':
                    self.fc1_in = self.cnn_output_size + (self.embedding_dim * self.tok_seq_len)
                elif self.inputs=='text':
                    self.fc1_in = self.embedding_dim * self.tok_seq_len

                self.fc1_out = self.bottleneck_feats # TODO make the same as below

                self.fc1 = nn.Linear(self.fc1_in, self.fc1_out)#,dropout=self.dropout)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(self.fc1_out, self.num_classes)

            else:
                #########################################
                # CNN-only version of my model        ###
                #########################################
                if self.inputs=='speech':
                    self.fc1_in = self.out_channels
                elif self.inputs=='both':
                    self.fc1_in = self.out_channels + self.embedding_dim
                elif self.inputs=='text':
                    self.fc1_in = self.embedding_dim

                self.fc1 = nn.Linear(self.fc1_in,self.bottleneck_feats) # TODO make the same as above
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(p=self.dropout)
                self.fc2 = nn.Linear(self.bottleneck_feats,self.num_classes)

    def convolve_timestamps(self,timestamps):
        '''
        Given a tensor of timestamps that correspond to where in the original time signal the token breaks come,
        calculate which frame in the encoder output will correspond to the token break.
        '''
        timestamps = torch.floor((timestamps + (2*self.padding[0]) - self.kernel1[0])/self.stride1[0]) + 1
        timestamps = torch.floor((timestamps + (2 * self.padding[0]) - self.kernel2[0]) / self.stride2[0]) + 1
        if self.cnn_layers>=3:
            timestamps = torch.floor((timestamps + (2 * self.padding[0]) - self.kernel2[0]) / self.stride2[0]) + 1
        if self.cnn_layers>=4:
            timestamps = torch.floor((timestamps + (2 * self.padding[0]) - self.kernel2[0]) / self.stride2[0]) + 1

        return timestamps

    def token_split(self,input,toktimes):
        #import pdb;pdb.set_trace() # TODO fix this so that the toktimes tensor also gets chopped up. Test with batch size of 1.
        #toktimes = [int(tim) for tim in toktimes.squeeze().tolist()]
        batch_size = input.shape[0]
        instances = []
        for j in range(batch_size):
            tokens = []
            instance = input[j:j+1,:,:]
            curr_toktimes = toktimes[j:j+1,:].squeeze()
            curr_toktimes = np.trim_zeros(np.array(curr_toktimes,dtype=np.int),trim='b').tolist()
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

        if self.postlstm_context:
            mark_focus = True
            tok_w_context = []
            for i,tok in enumerate(output):
                t_prev = torch.tensor(np.zeros(tok.shape),dtype=torch.float32).to(self.device) if i==0 else output[i-1]
                t_next = torch.tensor(np.zeros(tok.shape),dtype=torch.float32).to(self.device) if i==len(output)-1 else output[i+1]
                tok_w_context.append(torch.cat((t_prev,tok,t_next),dim=0))
            out = tok_w_context
        out = torch.cat(output,dim=0)
        return out

    def forward(self,x,text,toktimes,hidden):
        '''
        N: number of items in a batch
        C: number of channels
        W: number of frames in signal
        H: number of acoustic features in signal
        '''

        if self.inputs=='both' or self.inputs=='speech':

            if self.tok_level_pred:
                toktimes = self.convolve_timestamps(toktimes)

            # in: N x C x W x H
            x = self.conv(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))
            # in: N x C x W x H , where W is compressed and H=1

        if self.inputs=='both' or self.inputs=='text':
            embeddings = self.emb(text)
            embeddings = embeddings.permute(1,0,2)

        if self.tok_level_pred:
            ##############################################
            ## MY MODEL                                ###
            ##############################################

            if self.include_lstm:
                ################################
                ### CNN + LSTM                ##
                ################################
                #TOKENIZE_FIRST = True # This is the switch to toggle between doing LSTM -> tok vs tok -> LSTM. Not in config file yet.
                #if TOKENIZE_FIRST:
                if self.inputs=='both' or self.inputs=='speech':
                    x = x.squeeze(dim=-1)  # IN: N x C x W x H (where H=1) OUT: N x C x W
                    x = self.token_split(x, toktimes)
                    x = x.permute(1,0,2) # Comes out of tokens with dims: batch, seq_len, channels. Need seq_len, batch, channels
                if self.inputs=='both':
                    x = torch.cat([embeddings,x],dim=2)
                elif self.inputs=='text':
                    x = embeddings

                x,hidden = self.lstm(x,hidden) # In: seq_len, batch, channels. Out: seq_len, batch, hidden*2
                x = self.fc(x) # In: seq_len, batch, hidden*2. Out: seq_len, batch, num_classes
                return x,hidden
                """
                else:
                    # TODO make it possible to incorporate text here (maybe)
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
                """


            else:
                ################################
                ### CNN ONLY                  ##
                ################################
                if self.inputs=='speech' or self.inputs=='both':
                    x = x.squeeze(dim=-1)  # IN: N x C x W x H (where H=1) OUT: N x C x W
                    x = self.token_split(x, toktimes)
                    x = x.permute(1,0,2)  # Comes out of tokens with dims: batch, seq_len, channels. Need seq_len, batch, channels
                else:
                    x = embeddings

                if self.inputs=='both':
                    x = torch.cat([x,embeddings],dim=2)

                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x,hidden

        else:
            ########################################################
            # Dummy task: from sequence, predict one item's tag    #
            ########################################################
            if self.include_lstm:

                x = x.view(x.shape[0], x.shape[1], x.shape[2]).transpose(1, 2).transpose(0, 1).contiguous() # here: W x N x C
                x, hidden = self.lstm(x.view(x.shape[0], x.shape[1], x.shape[2]), hidden)
                # here: W x N x lstm_hidden_size
                #x = x[-1,:,:] # TAKE LAST TIMESTEP # here: 1 x N x lstm_hidden_size
                x = torch.mean(x,0)
                x = self.fc(x.view(1, x.shape[0], self.lin_input_size))  # in 1 x N? x lstm_hidden_size
                return x,hidden

            elif not self.include_lstm:
                ##############################################
                ## REPLICATION OF STEHWIEN ET AL.          ###
                ##############################################
                if self.inputs=='speech':
                    x = self.maxpool(x.view(x.shape[0], x.shape[1], x.shape[2]))
                elif self.inputs=='both':
                    # TODO make this path work the rest of the way
                    embeddings = embeddings.permute(1, 0, 2)  # TODO make less inefficient
                    print(x.shape)
                    x = self.maxpool(x.view(x.shape[0], x.shape[1], x.shape[2]))
                    print(x.shape)
                    import pdb;pdb.set_trace()
                    x = torch.cat([x,embeddings],dim=2) # TODO test
                elif self.inputs=='text':
                    embeddings = embeddings.permute(1, 0, 2)  # TODO make less inefficient
                    x = embeddings
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
