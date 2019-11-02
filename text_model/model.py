import pickle
import os
from torch import nn
import torch
import pandas as pd
import yaml
from utils import load_data,BatchWrapper,to_ints,load_vectors,make_batches,load_libri_data,plot_results,load_burnc_data
from evaluate import evaluate,last_only_evaluate
import numpy as np
import sys


if len(sys.argv) < 2:
    config = 'conf/burnc.yaml'
else:
    config = sys.argv[1]

with open(config,'r') as f:
    cfg = yaml.load(f,yaml.FullLoader)
torch.manual_seed(0)


print_dims = cfg['print_dims']
print_every = int(cfg['print_every'])
eval_every = int(cfg['eval_every'])
train_ratio = float(cfg['train_ratio'])
dev_ratio = float(cfg['dev_ratio'])

# hyperparameters
batch_size = int(cfg['batch_size'])
bidirectional = cfg['bidirectional']
learning_rate = float(cfg['learning_rate'])
embedding_dim = int(cfg['embedding_dim'])
hidden_size = int(cfg['hidden_size'])
use_pretrained = cfg['use_pretrained']
max_len = int(cfg['max_len'])
datasource = cfg['datasource']
vocab_size = int(cfg['vocab_size'])
num_epochs = int(cfg['num_epochs'])
num_layers = int(cfg['num_layers'])
dropout = float(cfg['dropout'])

# Filenames
datafile = cfg['datafile']
glove_path = cfg['glove_path']
model_name = cfg['model_name']
results_path = cfg['results_path']
results_file = '{}/{}.txt'.format(results_path,model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LOAD THE DATA

if datasource == 'SWBDNXT' or datasource == 'SWBDNXT_UTT':

    data = load_data(datafile,shuffle=True,max_len=max_len)

    train_idx = int(train_ratio*len(data))
    dev_idx = int((train_ratio+dev_ratio)*len(data))

    train_data = data[:train_idx]
    dev_data = data[train_idx:dev_idx]
    test_data = data[dev_idx:]

elif datasource == 'LIBRI':

    libritrain = '../data/libri/train_360.txt'
    libridev = '../data/libri/dev.txt'
    train_data = load_libri_data(libritrain,shuffle=True,max_len=max_len)
    dev_data = load_libri_data(libridev,shuffle=True,max_len=max_len)

elif datasource == 'BURNC':

    burncfeats = datafile
    data = load_burnc_data(burncfeats)

    train_idx = int(train_ratio*len(data))
    dev_idx = int((train_ratio+dev_ratio)*len(data))

    train_data = data[:train_idx]
    dev_data = data[train_idx:dev_idx]
    test_data = data[dev_idx:]

else:
    print("NO DATA SOURCE GIVEN")

X_train,Y_train,wd_to_i,i_to_wd = to_ints(train_data,vocab_size)
X_dev,Y_dev,_,_ = to_ints(dev_data,vocab_size,wd_to_i,i_to_wd)

# LOAD VECTORS

if use_pretrained:
    vec_dict_pkl = '../data/emb/50d-dict.pkl'
    if os.path.exists(vec_dict_pkl):
        with open(vec_dict_pkl,'rb') as f:
            i_to_vec = pickle.load(f)
    else:
        i_to_vec = load_vectors(glove_path, wd_to_i)
        with open(vec_dict_pkl, 'wb') as f:
            pickle.dump(i_to_vec, f)

    words_found = 0
    weights_matrix = np.zeros((vocab_size+2, embedding_dim))
    for i in i_to_wd:
        try:
            weights_matrix[i] = i_to_vec[i]
            words_found += 1
        except:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))

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
        if print_dims:
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

model = BiLSTM(batch_size=batch_size,
               vocab_size=vocab_size+2,
               tagset_size=2,
               bidirectional=bidirectional,
               lstm_layers=num_layers,
               embedding_dim=embedding_dim,
               hidden_size=hidden_size,
               use_pretrained=use_pretrained,
               dropout=dropout)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

# Add pre-training stats to output:
train_losses.append(0)
if not datasource == 'LIBRI':
    #_, train_acc, _ = evaluate(X_train, Y_train_str, model, device)
    train_acc = last_only_evaluate(X_train, Y_train_str, model, device)
    train_accs.append(train_acc)
else:  # Don't do train acc every time for bigger datasets than SWBDNXT
    train_accs.append(0)
#_, dev_acc, _ = evaluate(X_dev, Y_dev_str, model, device)
dev_acc = last_only_evaluate(X_dev, Y_dev_str, model, device)
dev_accs.append(dev_acc)
timesteps.append(0)

for epoch in range(num_epochs):
    # BATCH DATA
    X_train_batches, Y_train_batches = make_batches(X_train, Y_train, batch_size, device)

    print("TRAIN================================================================================================")
    for i in range(len(X_train_batches)):

        input, labels = X_train_batches[i], Y_train_batches[i]

        if not (list(input.shape)[0] == 0):

            timestep += 1
            model.zero_grad()

            hidden = model.init_hidden()
            tag_scores,_ = model(input,hidden)
            #print('pred:',tag_scores.view(labels.shape[0],labels.shape[1]))
            #print('true:',labels)
            loss = loss_fn(tag_scores.view(labels.shape[0],labels.shape[1]), labels.float())
            recent_losses.append(loss.detach())
            if len(recent_losses) > 50:
                recent_losses = recent_losses[1:]

            loss.backward()
            optimizer.step()


            if i % print_every == 1:
                avg_loss = sum(recent_losses)/len(recent_losses)
                print("Epoch: %s Step: %s Loss: %s"%(epoch,i,avg_loss.item())) # TODO could my loss calculation be deceiving?

            if i % eval_every == 1:
                train_loss = (sum(recent_losses)/len(recent_losses)).item()
                train_losses.append(train_loss)
                if not datasource == 'LIBRI':
                    #_,train_acc,_ = evaluate(X_train, Y_train_str, model,device)
                    train_acc = last_only_evaluate(X_train, Y_train_str, model, device)
                    train_accs.append(train_acc)
                else: # Don't do train acc every time for bigger datasets than SWBDNXT
                    train_accs.append(0)
                dev_acc = last_only_evaluate(X_dev, Y_dev_str, model,device)
                dev_accs.append(dev_acc)
                timesteps.append(timestep)

train_losses = pd.Series(train_losses)
train_accs = pd.Series(train_accs)
dev_accs = pd.Series(dev_accs)
train_steps = pd.Series(timesteps)


plot_results(train_losses,train_accs,dev_accs,train_steps,model_name,results_path)


print("==============================================")
print("==============================================")
print('After training, train:')
last_only_evaluate(X_train,Y_train_str,model,device)


print('After training, dev: ')
last_only_evaluate(X_dev, Y_dev_str,model,device)#,True)









