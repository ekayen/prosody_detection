# Get words in one-hot format
# Build model
from torch import nn
import torch
from torchtext import data,datasets
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils import load_data,BatchWrapper,to_ints
import numpy as np
from seqeval.metrics import accuracy_score, classification_report,f1_score

PRINT_DIMS = True
PRINT_EVERY = 1000
EVAL_EVERY = 2000

# Hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 3
BIDIRECTIONAL = True
LEARNING_RATE = 0.005
TRAIN_RATIO = 0.6
DEV_RATIO = 0.2
VOCAB_SIZE = 4000 # TODO actually restrict to high-freq vocab items
SOFTMAX_DIM = 2


datafile = '../data/all_acc.txt'
#datafile = '../data/mac_morpho/all.txt'
modelfile = 'model.pt'


# LOAD THE DATA

data = load_data(datafile,shuffle=True)

train_idx = int(TRAIN_RATIO*len(data))
dev_idx = int((TRAIN_RATIO+DEV_RATIO)*len(data))

train_data = data[:train_idx]
dev_data = data[train_idx:dev_idx]
test_data = data[dev_idx:]

X_train,Y_train,wd_to_i,i_to_wd = to_ints(train_data,VOCAB_SIZE) # TODO fix so that only train data ends up in vocab.
X_dev,Y_dev,_,_ = to_ints(dev_data,VOCAB_SIZE,wd_to_i,i_to_wd)
X_test,Y_test,_,_ = to_ints(test_data,VOCAB_SIZE,wd_to_i,i_to_wd)

#import pdb;pdb.set_trace()

# BUILD THE MODEL

class BiLSTM(nn.Module):
    # LSTM: (embedding_dim, hidden_size)
    # OUTPUT = (hidden size, tagset_size),
    # SOFTMAX over dimension 2 (I am not sure if this is right)
    def __init__(self, batch_size, vocab_size, tagset_size, embedding_dim=100, hidden_size=128, lstm_layers=1, bidirectional = True):

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
        self.softmax = nn.Softmax(dim=SOFTMAX_DIM)

    def forward(self,sent,hidden):
        self.seq_len = len(sent)# TODO change for real batching
        #import pdb;pdb.set_trace()
        embeds = self.embedding(sent)
        lstm_out, hidden = self.lstm(embeds.view(self.seq_len, self.batch_size, -1), hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.softmax(tag_space) #F.log_softmax(tag_space, dim=1)
        if PRINT_DIMS:

            print(embeds)
            print('embeds.shape', embeds.shape)
            print(lstm_out)
            print('lstm_out.shape', lstm_out.shape)
            print(tag_space)
            print('tag_space dims',tag_space.shape)
            print(tag_scores)
            print('tag_scores dims',tag_scores.shape)
            print(hidden)
            import pdb;pdb.set_trace()
        return tag_scores,hidden
        #return tag_space, hidden

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

# INSTANTIATE THE MODEL

model = BiLSTM(batch_size=BATCH_SIZE,vocab_size=VOCAB_SIZE+2,tagset_size=2,bidirectional=BIDIRECTIONAL)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# DEFINE EVAL FUNCTION

def evaluate(X, Y):

    y_pred = []

    with torch.no_grad():
        for i in range(len(X)):
            input = X[i]
            #input = torch.tensor([i if i in i_to_wd else wd_to_i['<UNK>'] for i in input])
            if not (list(input.shape)[0] == 0):
                hidden = model.init_hidden()
                tag_scores, _ = model(input, hidden)
                pred = np.squeeze(np.argmax(tag_scores, axis=-1)).tolist() # TODO could this be wrong? Almost certainly yes.
                if type(pred) is int:
                    pred = [pred]
                pred = [str(j) for j in pred]
                y_pred.append(pred)

    #import pdb;pdb.set_trace()
    print('Evaluation:')
    print('F1:',f1_score(Y, y_pred))
    print('Acc',accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

# Before training, evaluate
print('Before training, train:')
Y_train_str = [[str(i) for i in y.tolist()] for y in Y_train]
#evaluate(X_train,Y_train_str)


print('Before training, dev:')
Y_dev_str = [[str(i) for i in y.tolist()] for y in Y_dev]
#evaluate(X_dev,Y_dev_str)

#import pdb;pdb.set_trace()

# TRAIN

total_loss = 0
#loss_divisor = 0
for epoch in range(NUM_EPOCHS):
    for i in range(len(X_train)):

        model.zero_grad()
        input,labels = X_train[i],Y_train[i]
        input = torch.tensor([i if i in i_to_wd else wd_to_i['<UNK>'] for i in input])
        if not (list(input.shape)[0] == 0):

            hidden = model.init_hidden()

            tag_scores,_ = model(input,hidden)

            loss = loss_fn(tag_scores.view(model.seq_len,-1),labels)
            total_loss += loss
#            loss_divisor += model.seq_len
            loss.backward()
            optimizer.step()
            if i in range(10) and epoch==0:
                print('Instance number ',i)
                print(model.seq_len,'tokens')
                print(loss)
            if i % PRINT_EVERY == 1:
                print("Epoch: %s Step: %s Loss: %s"%(epoch,i,(total_loss/(i+(epoch*len(X_train)))).item())) # TODO could my loss calculation be deceiving? I definitely need to not just divide by i once I do more than one epoch
            if i % EVAL_EVERY == 1:
                evaluate(X_dev,Y_dev_str)

print('After training, train:')
evaluate(X_train,Y_train_str)


print('After training, dev: ')
evaluate(X_dev, Y_dev_str)















"""
# One forward pass to test dims:
# This would go right before the training loop
with torch.no_grad():
    input,labels = X[5],Y[5]
    print('seq_len',X[5].shape)
    hidden = model.init_hidden()
    print("hidden0: ", hidden[0].shape)
    tag_scores = model(input,hidden)
    print("labels dims",labels.shape)
import pdb;pdb.set_trace()
"""




# Use torchtext to numericalize the text and pad the seqs

"""
# This would go at the beginning, during the data loading step
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

