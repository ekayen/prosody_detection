import pickle
import os
from torch import nn
import torch
import pandas as pd
import yaml
from utils import *
from evaluate import *
import numpy as np
import sys
import random

if len(sys.argv) < 2:
    config = 'conf/burnc.yaml'
else:
    config = sys.argv[1]

with open(config,'r') as f:
    cfg = yaml.load(f,yaml.FullLoader)

seed = cfg['seed']

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if cfg['include_lstm']:
    model_type = 'lstm'
else:
    model_type = 'simpleff'

#print_preds = cfg['print_preds']
#print_dims = cfg['print_dims']
#print_every = int(cfg['print_every'])
#eval_every = int(cfg['eval_every'])
#train_ratio = float(cfg['train_ratio'])
#dev_ratio = float(cfg['dev_ratio'])

# hyperparameters
batch_size = int(cfg['batch_size'])
bidirectional = cfg['bidirectional']
learning_rate = float(cfg['learning_rate'])
embedding_dim = int(cfg['embedding_dim'])
hidden_size = int(cfg['hidden_size'])
use_pretrained = cfg['use_pretrained']
max_len = int(cfg['max_len'])
#datasource = cfg['datasource']
vocab_size = int(cfg['vocab_size'])
num_epochs = int(cfg['num_epochs'])
lstm_layers = int(cfg['lstm_layers']) #num_layers = int(cfg['num_layers'])
dropout = float(cfg['dropout'])

# Filenames
#datafile = cfg['datafile']
glove_path = cfg['glove_path']
model_name = cfg['model_name']
results_path = cfg['results_path']
results_file = '{}/{}.txt'.format(results_path,model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# BUILD THE MODEL

class FFModel(nn.Module):
    def __init__(self,embedding_dim,vocab_size,bottleneck_feats,use_pretrained=True,window_size=3,num_classes=1):
        super(FFModel, self).__init__()
        self.bottleneck_feats = bottleneck_feats
        self.window_size = window_size
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size+2,embedding_dim)
        if use_pretrained:
            self.embedding.load_state_dict({'weight': weights_matrix})
            self.embedding.weight.requires_grad = False
        self.fc1 = nn.Linear(embedding_dim*self.window_size,self.bottleneck_feats)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.bottleneck_feats,self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.embedding(x)
        batch_dim = x.shape[1]
        x = x.transpose(0,1).contiguous().view(batch_dim,-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


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
                 nontrainable=False):

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
        self.use_pretrained = use_pretrained

        # layers:
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if self.use_pretrained:
            self.embedding.load_state_dict({'weight': weights_matrix})
            if nontrainable:
                self.embedding.weight.requires_grad = False
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
        tag_scores = tag_space
        #tag_scores = self.sigmoid(tag_space)
        print_dims = False
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

set_seeds(seed)

# LOAD THE DATA

with open(cfg['all_data'], 'rb') as f:
    data_dict = pickle.load(f)
with open(cfg['datasplit'].replace('yaml','vocab'),'rb') as f:
    vocab_dict = pickle.load(f)
random.seed(seed)

def truncate_dicts(vocab_dict,vocab_size):
    i2w = {}
    w2i = {}
    for i in range(vocab_size+2):
        w = vocab_dict['i2w'][i]
        i2w[i] = w
        w2i[w] = i
    return w2i,i2w

w2i,i2w = truncate_dicts(vocab_dict,vocab_size)

trainset = BurncDatasetText(cfg, data_dict, w2i, vocab_size=vocab_size,
                            mode='train', datasplit=cfg['datasplit'])
devset = BurncDatasetText(cfg, data_dict, w2i, vocab_size=vocab_size,
                          mode='dev',datasplit=cfg['datasplit'])

# LOAD VECTORS
words_found = 0
if use_pretrained:
    vec_dict_pkl = '../data/emb/100d-dict.pkl'
    if os.path.exists(vec_dict_pkl) and embedding_dim==100:
        with open(vec_dict_pkl,'rb') as f:
            i_to_vec = pickle.load(f)
    else:
        i_to_vec = load_vectors(glove_path,  w2i)
        with open(vec_dict_pkl, 'wb') as f:
            pickle.dump(i_to_vec, f)

    weights_matrix = np.zeros((vocab_size+2, embedding_dim))
    for i in i2w:
        try:
            weights_matrix[i] = i_to_vec[i]
            words_found += 1
        except:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))

    weights_matrix = torch.tensor(weights_matrix)

# INSTANTIATE THE MODEL

set_seeds(seed)

if model_type == 'lstm':
    print('complex model')
    model = BiLSTM(batch_size=batch_size,
                   vocab_size=vocab_size+2,
                   tagset_size=2,
                   bidirectional=bidirectional,
                   lstm_layers=lstm_layers,
                   embedding_dim=embedding_dim,
                   hidden_size=hidden_size,
                   use_pretrained=use_pretrained,
                   dropout=dropout)

elif model_type == 'simpleff':
    print('simple model')
    bottleneck_feats = cfg['bottleneck_feats']
    model = FFModel(embedding_dim,vocab_size,bottleneck_feats,use_pretrained=use_pretrained)


#loss_fn = nn.BCELoss()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.to(device)


# TRAIN
recent_losses = []
timestep = 0
plot_data = {
        'time': [],
        'loss': [],
        'train_acc': [],
        'dev_acc': [],
    }

print_every = 100
eval_every = 100

"""
# Add pre-training stats to output:
train_losses.append(0)
if not datasource == 'LIBRI':
    _, train_acc, _ = evaluate(X_train, Y_train_str, model, device,i_to_wd,model_type=model_type,print_preds=print_preds,timestep=0)
    #train_acc = last_only_evaluate(X_train, Y_train_str, model, device)
    train_accs.append(train_acc)
else:  # Don't do train acc every time for bigger datasets than SWBDNXT
    train_accs.append(0)
_, dev_acc, _ = evaluate(X_dev, Y_dev_str, model, device,i_to_wd,model_type=model_type,print_preds=print_preds,timestep=0)
#dev_acc = last_only_evaluate(X_dev, Y_dev_str, model, device)
dev_accs.append(dev_acc)
timesteps.append(0)
"""

#true_labels = []

traingen = data.DataLoader(trainset, **cfg['train_params'])

for epoch in range(num_epochs):

    print("TRAIN================================================================================================")
    for id, batch, labels in traingen:
        model.train()
        #input, labels = X_train_batches[i], Y_train_batches[i]
        if cfg['segmentation']=='tokens':
            labels = labels.view(labels.shape[0],1)
        input = batch[0].transpose(0,1).to(device)
        labels = labels.transpose(0,1).to(device)

        curr_bat_size = input.shape[1]
        #import pdb;pdb.set_trace()
        if not (list(input.shape)[0] == 0):
            """
            if epoch==0:
                true_labels.append((input,labels))
            """
            timestep += 1
            model.zero_grad()

            if cfg['include_lstm']:
                hidden = model.init_hidden(curr_bat_size)
                tag_scores,_ = model(input,hidden)
            else:
                tag_scores = model(input)


            loss = loss_fn(tag_scores.view(labels.shape[0],labels.shape[1]), labels.float())
            recent_losses.append(loss.detach())
            if len(recent_losses) > 50:
                recent_losses = recent_losses[1:]

            loss.backward()
            optimizer.step()

            avg_loss = sum(recent_losses)/len(recent_losses)

    print("Epoch: %s Loss: %s"%(epoch,avg_loss.item()))
    plot_data['time'].append(epoch)

    train_loss = (sum(recent_losses)/len(recent_losses)).item()
    plot_data['loss'].append(train_loss)

    train_results = evaluate(trainset, cfg['train_params'], model, device,
                             tok_level_pred=cfg['tok_level_pred'], noisy=True,text_only=True)
    plot_data['train_acc'].append(train_results[0])
    dev_results = evaluate(devset, cfg['eval_params'], model, device,
                             tok_level_pred=cfg['tok_level_pred'], noisy=True,text_only=True)
    plot_data['dev_acc'].append(dev_results[0])

"""
    with open('../data/burnc/train_w_true_labels.tsv','w') as f:
        for x,y in true_labels:
            print(x)
            print(y)
            x = x.transpose(0, 1).tolist()
            y = y.transpose(0, 1).tolist()
            x = [list(np.trim_zeros(np.array(utt))) for utt in x]
            tmp = []
            for i,utt in enumerate(y):
                tmp.append(utt[:len(x[i])])
            y = tmp
            for words,labels in zip(x,y):
                words = [i_to_wd[i] for i in words]
                labels = [str(i) for i in labels]
                f.write(' '.join(words)+'\t'+' '.join(labels)+'\n')
    """

plot_results(plot_data,model_name,results_path)


print("==============================================")
print("==============================================")
print('After training, train:')
#evaluate(X_train,Y_train_str,model,device,i_to_wd,model_type=model_type,print_preds=print_preds)

print('After training, dev: ')
#evaluate(X_dev, Y_dev_str,model,device,i_to_wd,model_type=model_type,print_preds=print_preds)









