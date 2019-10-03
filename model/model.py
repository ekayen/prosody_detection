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

from utils import load_data,BatchWrapper,to_ints,load_vectors,make_batches
import numpy as np
from seqeval.metrics import accuracy_score, classification_report,f1_score

PRINT_DIMS = False
PRINT_EVERY = 50
EVAL_EVERY = 100
TRAIN_RATIO = 0.6
DEV_RATIO = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 15
NUM_LAYERS = 2
BIDIRECTIONAL = True
LEARNING_RATE = 0.001
VOCAB_SIZE = 4000
SOFTMAX_DIM = 2
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
DROPOUT = 0.5
USE_PRETRAINED = False
MAX_LEN = 80

datafile = '../data/all_acc.txt'
#glove_path = '../data/emb/glove.6B.50d.txt'
glove_path = '../data/emb/glove.6B.100d.txt'

import sys

model_name = sys.argv[1]
results_file = 'results/{}.txt'.format(model_name)
model_file = '{}.pt'.format(model_name)

# LOAD THE DATA

data = load_data(datafile,shuffle=True,max_len=MAX_LEN)

train_idx = int(TRAIN_RATIO*len(data))
dev_idx = int((TRAIN_RATIO+DEV_RATIO)*len(data))

train_data = data[:train_idx]
dev_data = data[train_idx:dev_idx]
test_data = data[dev_idx:]

X_train,Y_train,wd_to_i,i_to_wd = to_ints(train_data,VOCAB_SIZE)
X_dev,Y_dev,_,_ = to_ints(dev_data,VOCAB_SIZE,wd_to_i,i_to_wd)
X_test,Y_test,_,_ = to_ints(test_data,VOCAB_SIZE,wd_to_i,i_to_wd)


# LOAD VECTORS





# BUILD THE MODEL

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
            #self.hidden2tag = nn.Linear(2 * hidden_size, tagset_size) # TODO change this to 1, rather than 2
            self.hidden2tag = nn.Linear(2 * hidden_size, output_dim)
        else:
            #self.hidden2tag = nn.Linear(hidden_size, tagset_size)
            self.hidden2tag = nn.Linear(hidden_size, output_dim)
        #self.softmax = nn.Softmax(dim=SOFTMAX_DIM)
        self.sigmoid = nn.Sigmoid()

    def forward(self,sent,hidden): # TODO figure out batching
        self.seq_len = sent.shape[0]
        embeds = self.embedding(sent)#.view(self.seq_len,self.batch_size))
        #lstm_out, hidden = self.lstm(embeds.view(self.seq_len, self.batch_size, -1), hidden)
        lstm_out, hidden = self.lstm(embeds, hidden)
        tag_space = self.hidden2tag(lstm_out)
        #tag_scores = self.softmax(tag_space)
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
#loss_fn = nn.CrossEntropyLoss() # TODO change to binary crossentropy loss (maybe)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.to(device)

# DEFINE EVAL FUNCTION

def evaluate(X, Y,mdl):
    print("EVAL=================================================================================================")
    y_pred = []
    with torch.no_grad():
        for i in range(len(X)):
            #input = torch.tensor([i if i in i_to_wd else wd_to_i['<UNK>'] for i in input])
            input = X[i].to(device)
            eval_batch_size = 1

            if not (list(input.shape)[0] == 0):
                hidden = mdl.init_hidden(eval_batch_size)
                tag_scores, _ = mdl(input.view(len(input),eval_batch_size), hidden)

                #pred = np.squeeze(np.argmax(tag_scores, axis=-1)).tolist() # TODO could this be wrong? Almost certainly yes.

                pred = tag_scores.cpu()
                pred = np.where(pred>0.5,1,0)
                pred = np.squeeze(pred)
                pred = pred.tolist()
                if type(pred) is int:
                    pred = [pred]
                #pred = [[str(j) for j in i] for i in pred]
                pred = [str(j) for j in pred]

                #y_pred += pred
                y_pred.append(pred)
    print('Evaluation:')
    #import pdb;pdb.set_trace()
    #Y = np.concatenate([y.cpu().flatten() for y in Y]).tolist()
    #y_pred = np.concatenate([y.flatten() for y in y_pred]).tolist()
    #Y = [str(y) for y in Y]
    #y_pred = [str(y) for y in y_pred]

    f1 = f1_score(Y, y_pred)
    acc = accuracy_score(Y, y_pred)
    clss = classification_report(Y, y_pred)
    print('F1:',f1)
    print('Acc',acc)
    print(clss)
    return(f1,acc,clss)

Y_train_str = [[str(i) for i in y.tolist()] for y in Y_train]
#Y_train_batches_str = [[[str(i) for i in inst] for inst in batch.numpy()] for batch in Y_train_batches]


Y_dev_str = [[str(i) for i in y.tolist()] for y in Y_dev]
#Y_dev_batches_str = [[[str(i) for i in inst] for inst in batch.numpy()] for batch in Y_dev_batches]


"""
# Before training, evaluate on train data:
print('Before training, train:')
evaluate(X_train,Y_train_str,model)

print('Before training, evaluate on dev data:')
evaluate(X_dev,Y_dev_str,model)
"""
"""
def majority_baseline(X,Y):
    preds = []
    for x in X:
        tmp = np.zeros(x.shape)
        #tmp = np.random.randint(2,size=x.shape[0])
        tmp.tolist()
        tmp = [str(int(i)) for i in tmp]
        preds.append(tmp)
    print("Majority baseline:")
    print('F1:',f1_score(Y, preds))
    print('Acc',accuracy_score(Y, preds))
    print(classification_report(Y, preds))

majority_baseline(X_dev,Y_dev_str)

import pdb;pdb.set_trace()
"""

train_losses = []
train_accs = []
dev_accs = []

# TRAIN
recent_losses = []

timestep = 0
timesteps = []
for epoch in range(NUM_EPOCHS):
    # BATCH DATA
    X_train_batches, Y_train_batches = make_batches(X_train, Y_train, BATCH_SIZE, device)

    print("TRAIN================================================================================================")
    for i in range(len(X_train_batches)):

        #input,labels = X_train[i],Y_train[i]
        input, labels = X_train_batches[i], Y_train_batches[i]

        if not (list(input.shape)[0] == 0):

            timestep += 1
            model.zero_grad()

            hidden = model.init_hidden()
            tag_scores,_ = model(input,hidden)

            #loss = loss_fn(tag_scores.view(model.seq_len,-1),labels)
            loss = loss_fn(tag_scores.view(labels.shape[0],labels.shape[1]), labels.float())
            recent_losses.append(loss.detach())
            if len(recent_losses) > 50:
                recent_losses = recent_losses[1:]

            loss.backward()
            optimizer.step()


            if i % PRINT_EVERY == 1:
                avg_loss = sum(recent_losses)/len(recent_losses)
                print("Epoch: %s Step: %s Loss: %s"%(epoch,i,avg_loss.item())) # TODO could my loss calculation be deceiving?
                """ 
                # Print some weights to see if they move
                for name,param in model.named_parameters():
                    if name == 'lstm.weight_ih_l0':
                        print(name,param)
                """
            if i % EVAL_EVERY == 1:
                #    evaluate(X_dev,Y_dev_str,model)
                train_loss = (sum(recent_losses)/len(recent_losses)).item()
                train_losses.append(train_loss)
                _, train_acc, _ = evaluate(X_train, Y_train_str, model)
                train_accs.append(train_acc)
                _, dev_acc, _ = evaluate(X_dev, Y_dev_str, model)
                dev_accs.append(dev_acc)
                timesteps.append(timestep)

train_losses = pd.Series(train_losses)
train_accs = pd.Series(train_accs)
dev_accs = pd.Series(dev_accs)
train_steps = pd.Series(timesteps)

df = pd.DataFrame(dict(train_steps=train_steps,
                       train_losses=train_losses,
                       train_accs=train_accs,
                       dev_accs=dev_accs))

with open("tmp.pkl",'wb') as f:
    pickle.dump(df,f)

ax = plt.gca()
df.plot(kind='line',x='train_steps',y='train_losses',ax=ax)
df.plot(kind='line',x='train_steps',y='train_accs', color='red', ax=ax)
df.plot(kind='line',x='train_steps',y='dev_accs', color='green', ax=ax)


plt.savefig('results/{}.png'.format(model_name))
plt.show()
df.to_csv('results/{}.tsv'.format(model_name),sep='\t')

print("==============================================")
print("==============================================")
print('After training, train:')
evaluate(X_train,Y_train_str,model)


print('After training, dev: ')
evaluate(X_dev, Y_dev_str,model)









