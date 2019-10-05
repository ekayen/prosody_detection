# Get words in one-hot format
# Build model
import pickle
import os
from torch import nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
torch.manual_seed(0)

import torch.nn.functional as F

from utils import load_data,BatchWrapper,to_ints,load_vectors,make_batches,load_libri_data,plot_results
from evaluate import evaluate
import numpy as np

PRINT_DIMS = False
PRINT_EVERY = 50
EVAL_EVERY = 100
TRAIN_RATIO = 0.6
DEV_RATIO = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
BIDIRECTIONAL = True
LEARNING_RATE = 0.001
SOFTMAX_DIM = 2
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
USE_PRETRAINED = False
MAX_LEN = 80
DATASOURCE = 'SWBDNXT'


datafile = '../data/all_acc.txt'
#glove_path = '../data/emb/glove.6B.50d.txt'
glove_path = '../data/emb/glove.6B.100d.txt'

import sys

model_name = sys.argv[1]
results_file = 'results/{}.txt'.format(model_name)
model_file = '{}.pt'.format(model_name)

# LOAD THE DATA

if DATASOURCE == 'SWBDNXT':

    VOCAB_SIZE = 4000
    NUM_EPOCHS = 8
    NUM_LAYERS = 2
    DROPOUT = 0.5

    data = load_data(datafile,shuffle=True,max_len=MAX_LEN)

    train_idx = int(TRAIN_RATIO*len(data))
    dev_idx = int((TRAIN_RATIO+DEV_RATIO)*len(data))

    train_data = data[:train_idx]
    dev_data = data[train_idx:dev_idx]
    test_data = data[dev_idx:]


elif DATASOURCE == 'LIBRI':

    VOCAB_SIZE = 35000
    NUM_EPOCHS = 5
    NUM_LAYERS = 3
    DROPOUT = 0.2

    libritrain = '../data/libri/train_360.txt'
    libridev = '../data/libri/dev.txt'
    train_data = load_libri_data(libritrain,shuffle=True,max_len=MAX_LEN)
    dev_data = load_libri_data(libridev,shuffle=True,max_len=MAX_LEN)

else:
    print("NO DATA SOURCE GIVEN")

X_train,Y_train,wd_to_i,i_to_wd = to_ints(train_data,VOCAB_SIZE)
X_dev,Y_dev,_,_ = to_ints(dev_data,VOCAB_SIZE,wd_to_i,i_to_wd)

# LOAD VECTORS

if USE_PRETRAINED:
    vec_dict_pkl = '../data/emb/50d-dict.pkl'
    if os.path.exists(vec_dict_pkl):
        with open(vec_dict_pkl,'rb') as f:
            i_to_vec = pickle.load(f)
    else:
        i_to_vec = load_vectors(glove_path, wd_to_i)
        with open(vec_dict_pkl, 'wb') as f:
            pickle.dump(i_to_vec, f)

    words_found = 0
    weights_matrix = np.zeros((VOCAB_SIZE+2, EMBEDDING_DIM))
    for i in i_to_wd:
        try:
            weights_matrix[i] = i_to_vec[i]
            words_found += 1
        except:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM, ))

    weights_matrix = torch.tensor(weights_matrix)


# BUILD THE MODEL

class BiLSTM(nn.Module):
    # LSTM: (embedding_dim, hidden_size)
    # OUTPUT = (hidden size, tagset_size),
    # SOFTMAX over dimension 2 (I am not sure if this is right)
    def __init__(self,
                 batch_size,
                 vocab_size,
                 tagset_size,
                 embedding_dim=100,
                 hidden_size=128,
                 lstm_layers=2,
                 bidirectional=True,
                 output_dim=1,
                 dropout=0.5,
                 use_pretrained=False,
                 nontrainable = False):

        super(BiLSTM,self).__init__()

        # hparams:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        self.dropout = dropout

        # layers:
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        emb_layer = nn.Embedding(vocab_size, embedding_dim)
        if use_pretrained:
            emb_layer.load_state_dict({'weight': weights_matrix})
            if nontrainable:
                emb_layer.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size,
                            bidirectional=self.bidirectional,
                            num_layers=lstm_layers,
                            dropout=self.dropout)
        if self.bidirectional:
            self.hidden2tag = nn.Linear(2 * hidden_size, output_dim)
        else:
            self.hidden2tag = nn.Linear(hidden_size, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self,sent,hidden):
        self.seq_len = sent.shape[0]
        embeds = self.embedding(sent)
        lstm_out, hidden = self.lstm(embeds, hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.sigmoid(tag_space)
        if PRINT_DIMS:
            print('sent.shape',sent.shape)
            print('embeds.shape', embeds.shape)
            print('lstm_out.shape', lstm_out.shape)
            print('tag_space dims',tag_space.shape)
            print('tag_scores dims',tag_scores.shape)
            print('===============================================')

        return tag_scores,hidden

    def init_hidden(self,batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        if self.bidirectional:
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(device)
            c0 = torch.zeros(self.lstm_layers*2, batch_size, self.hidden_size).requires_grad_().to(device)
        else:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).requires_grad_().to(device)

        return (h0,c0)

# INSTANTIATE THE MODEL

model = BiLSTM(batch_size=BATCH_SIZE,
               vocab_size=VOCAB_SIZE+2,
               tagset_size=2,
               bidirectional=BIDIRECTIONAL,
               lstm_layers=NUM_LAYERS,
               embedding_dim=EMBEDDING_DIM,
               hidden_size=HIDDEN_SIZE,
               use_pretrained=USE_PRETRAINED,
               dropout=DROPOUT)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.to(device)

# Process data to work with eval library
Y_train_str = [[str(i) for i in y.tolist()] for y in Y_train]
Y_dev_str = [[str(i) for i in y.tolist()] for y in Y_dev]


# TRAIN
recent_losses = []
timestep = 0
timesteps = []

train_losses = []
train_accs = []
dev_accs = []

for epoch in range(NUM_EPOCHS):
    # BATCH DATA
    X_train_batches, Y_train_batches = make_batches(X_train, Y_train, BATCH_SIZE, device)

    print("TRAIN================================================================================================")
    for i in range(len(X_train_batches)):

        input, labels = X_train_batches[i], Y_train_batches[i]

        if not (list(input.shape)[0] == 0):

            timestep += 1
            model.zero_grad()

            hidden = model.init_hidden()
            tag_scores,_ = model(input,hidden)

            loss = loss_fn(tag_scores.view(labels.shape[0],labels.shape[1]), labels.float())
            recent_losses.append(loss.detach())
            if len(recent_losses) > 50:
                recent_losses = recent_losses[1:]

            loss.backward()
            optimizer.step()


            if i % PRINT_EVERY == 1:
                avg_loss = sum(recent_losses)/len(recent_losses)
                print("Epoch: %s Step: %s Loss: %s"%(epoch,i,avg_loss.item())) # TODO could my loss calculation be deceiving?

            if i % EVAL_EVERY == 1:
                train_loss = (sum(recent_losses)/len(recent_losses)).item()
                train_losses.append(train_loss)
                if DATASOURCE == 'SWBDNXT':
                    _,train_acc,_ = evaluate(X_train, Y_train_str, model,device)
                    train_accs.append(train_acc)
                else: # Don't do train acc every time for bigger datasets than SWBDNXT
                    train_accs.append(0)
                _, dev_acc, _ = evaluate(X_dev, Y_dev_str, model,device)
                dev_accs.append(dev_acc)
                timesteps.append(timestep)

train_losses = pd.Series(train_losses)
train_accs = pd.Series(train_accs)
dev_accs = pd.Series(dev_accs)
train_steps = pd.Series(timesteps)


plot_results(train_losses,train_accs,dev_accs,train_steps,model_name)

print("==============================================")
print("==============================================")
print('After training, train:')
evaluate(X_train,Y_train_str,model)


print('After training, dev: ')
evaluate(X_dev, Y_dev_str,model)#,True)









