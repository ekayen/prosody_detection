# Get words in one-hot format
# Build model
from torch import nn
import torch
from torchtext import data,datasets
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils import load_data,BatchWrapper


# Use torchtext to numericalize the text and pad the seqs

text = data.Field(lower=True,batch_first=True)
labels = data.Field(is_target=True,batch_first=True)
accents = datasets.SequenceTaggingDataset(
    path='../train_data/all_acc.conll',
    fields=[('text', text),
            ('labels', labels)]
)
text.build_vocab(accents) #, min_freq=2)
labels.build_vocab(accents)
train_iter = data.BucketIterator(dataset=accents, batch_size=32,  sort_key=lambda x: len(x.text), device=-1)


batch = next(iter(train_iter))
sent = batch.text[0:1,]
sent = sent.tolist()[0]
txt = [text.vocab.itos[i] for i in sent]
lbl = batch.labels[0:1,]
lbl = lbl.tolist()[0]

import pdb;pdb.set_trace()
print("sanity check of one sentence:")
print(txt,lbl)


class LSTM(nn.module):
    # INPUT (batch_size, seq_len, vocab_size)
    # LSTM: (vocab_size(?), hidden_size)
    # OUTPUT = (hidden size, seq_len), all binary values
    # Keras version:
    # seq_len, batch_size
    # ___, seq_len
    def __init__(self, seq_len, batch_size, vocab_size, tagset_size, hidden_size=128, lstm_layers=1):
        super(LSTM,self).__init__()
        # hparams:
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        # layers:
        self.embedding = nn.Embedding(seq_len, hidden_size)  # Currently using one-hot instead of embeddings, but this is here for future
        self.lstm = nn.LSTM(vocab_size, hidden_size, bidirectional=True, num_layers=lstm_layers)
        self.hidden2tag = nn.Linear(hidden_size, tagset_size)

    def forward(self,input,hidden):
        # Or maybe just insert F.one_hot here?
        #x = self.embedding(input).view(1,1,-1)
        one_hots = F.one_hot(input) # Should have dims (batch_size, seq_len, vocab_size)
        lstm_out,hidden = self.lstm(one_hots.view(seq_len,1,-1),hidden) # Not sure why you might need that 'view' thing...
        tag_space= self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return output,hidden


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


# build forward fn
# build predict fn
# build eval fn
