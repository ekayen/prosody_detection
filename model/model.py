# Get words in one-hot format
# Build model
from torch import nn
import torch
from torchtext import data,datasets
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils import load_data,BatchWrapper,to_ints

BATCH_SIZE = 1
#datafile = '../train_data/all_acc.conll'
datafile = '../data/all_acc.txt'

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

data = load_data(datafile)

X,Y,wd_to_i,i_to_wd = to_ints(data)


class BiLSTM(nn.Module):
    # LSTM: (vocab_size(?), hidden_size)
    # OUTPUT = (hidden size, seq_len), all binary values
    # Keras version:
    # seq_len, batch_size
    # ___, seq_len
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
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, num_layers=lstm_layers)
        if self.bidirectional:
            self.hidden2tag = nn.Linear(2 * hidden_size, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_size, tagset_size)

    def forward(self,sent,hidden):
        embeds = self.embedding(sent)
        print(embeds.shape)
        lstm_out, hidden = self.lstm(embeds.view(len(sent),1,-1), hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores,hidden

    def init_hidden(self):
        if self.bidirectional:
            first_dim = self.lstm_layers*2
        else:
            first_dim = self.lstm_layers
        # Initialize hidden state with zeros
        h0 = torch.zeros(first_dim, self.batch_size, self.hidden_size).requires_grad_()
        print('h0',h0.shape)

        # Initialize cell state
        c0 = torch.zeros(first_dim, self.batch_size, self.hidden_size).requires_grad_()
        print('c0',c0.shape)
        #hidden = (torch.zeros(1, 1, self.hidden_size),
        #          torch.zeros(1, 1, self.hidden_size))
        return (h0,c0)

model = BiLSTM(batch_size=BATCH_SIZE,vocab_size=len(wd_to_i),tagset_size=2)


# One forward pass to test dims:
with torch.no_grad():
    input,labels = X[5],Y[5]
    hidden = model.init_hidden()
    tag_scores = model(input,hidden)
    print(tag_scores)