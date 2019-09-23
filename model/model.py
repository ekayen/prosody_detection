# Get words in one-hot format
# Build model
from torch import nn
import torch
from torchtext import data,datasets
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils import load_data,BatchWrapper,to_ints

PRINT_DIMS = False
PRINT_EVERY = 1000

# Hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 10
BIDIRECTIONAL = True
LEARNING_RATE = 0.01
TRAIN_RATIO = 0.6
DEV_RATIO = 0.2


#datafile = '../train_data/all_acc.conll'
#datafile = '../data/all_acc.txt'
datafile = '../data/mac_morpho/all.txt'

# Use torchtext to numericalize the text and pad the seqs

"""
text = data.Field(lower=True,batch_first=True)
labels = data.Field(is_target=True,batch_first=True)
accents = datasets.SequenceTaggingDataset(
    path='../train_data/all_acc.conll',
    fields=[('text', text),
            ('labels', labels)]
)
text.build_vocab(accents) #, min_freq=2)
labels.build_vocab(accents)
train_iter = data.BucketIterator(dataset=accents, batch_size=BATCH_SIZE,  sort_key=lambda x: len(x.text))
train_iter = BatchWrapper(train_iter,'text','labels')

batch = next(iter(train_iter))

sent = batch[0][0:1,]
sent = sent.tolist()[0]
txt = [text.vocab.itos[i] for i in sent]
lbl = batch[1][0:1,]
lbl = lbl.tolist()[0]


print("sanity check of one sentence:")
print(txt,lbl)
"""

data = load_data(datafile,shuffle=True)

X,Y,wd_to_i,i_to_wd = to_ints(data)

train_idx = int(TRAIN_RATIO*len(X))
dev_idx = int((TRAIN_RATIO+DEV_RATIO)*len(X))
X_train = X[:train_idx]
Y_train = Y[:train_idx]
X_dev = X[train_idx:dev_idx]
Y_dev = Y[train_idx:dev_idx]
X_test = X[dev_idx:]
Y_test = Y[dev_idx:]

class BiLSTM(nn.Module):
    # LSTM: (embedding_dim, hidden_size)
    # OUTPUT = (hidden size, tagset_size),
    # SOFTMAX over dimension 1 (I am not sure if this is right)
    def __init__(self, batch_size, vocab_size, tagset_size, embedding_dim=100, hidden_size=128, lstm_layers=1, bidirectional = BIDIRECTIONAL):

        super(BiLSTM,self).__init__()

        # hparams:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        #self.seq_len = seq_len

        # layers:
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Currently using one-hot instead of embeddings, but this is here for future
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=self.bidirectional, num_layers=lstm_layers)
        if self.bidirectional:
            self.hidden2tag = nn.Linear(2 * hidden_size, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_size, tagset_size)
        self.softmax = nn.LogSoftmax(dim=2) # TODO Check this

    def forward(self,sent,hidden):
        self.seq_len = len(sent)# TODO change for real batching
        embeds = self.embedding(sent)
        lstm_out, hidden = self.lstm(embeds.view(self.seq_len, self.batch_size, -1), hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.softmax(tag_space) #F.log_softmax(tag_space, dim=1)
        if PRINT_DIMS:
            print('embeds.shape',embeds.shape)
            print('lstm_out.shape',lstm_out.shape)
            print('tag_space dims',tag_space.shape)
            print('tag_scores dims',tag_scores.shape)
            import pdb;pdb.set_trace()
        return tag_scores,hidden

    def init_hidden(self):

        if self.bidirectional:
            first_dim = self.lstm_layers*2
        else:
            first_dim = self.lstm_layers

        # Initialize hidden state with zeros
        h0 = torch.zeros(first_dim, self.batch_size, self.hidden_size).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(first_dim, self.batch_size, self.hidden_size).requires_grad_()
        return (h0,c0)

model = BiLSTM(batch_size=BATCH_SIZE,vocab_size=len(wd_to_i),tagset_size=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

"""
# One forward pass to test dims:
with torch.no_grad():
    input,labels = X[5],Y[5]
    print('seq_len',X[5].shape)
    hidden = model.init_hidden()
    print("hidden0: ", hidden[0].shape)
    tag_scores = model(input,hidden)
    print("labels dims",labels.shape)
import pdb;pdb.set_trace()
"""

total_loss = 0
for epoch in range(NUM_EPOCHS):
    for i in range(len(X_train)):

        model.zero_grad()
        input,labels = X_train[i],Y_train[i]
        if not (list(input.shape)[0] == 0):

            hidden = model.init_hidden()

            tag_scores,_ = model(input,hidden)
            #import pdb; pdb.set_trace()
            loss = loss_fn(tag_scores.view(model.seq_len,-1),labels)
            total_loss += loss
            loss.backward()
            optimizer.step()
            if i % PRINT_EVERY == 0:
                print("Loss: ",total_loss/i)

import pdb;pdb.set_trace()

# EVALUATION

# Single example to check
with torch.no_grad():
    inputs = X_train[26]
    tag_scores = model(inputs)
    print(tag_scores[0])

with torch.no_grad():
    for i in range(len(X_dev)):
        input,labels = X_dev[i],Y_dev[i]


