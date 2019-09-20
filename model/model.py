# Get words in one-hot format
# Build model
from torch import nn
import torch
from torchtext import data,datasets
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


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

print("sanity check of one sentence:")
print(txt,lbl)


class LSTM(nn.module):
    # INPUT (batch_size, seq_len, vocab_size)
    # LSTM: (vocab_size(?), hidden_size)
    # OUTPUT = (hidden size, seq_len), all binary values
    # Keras version:
    # seq_len, batch_size
    # ___, seq_len
    def __init__(self, seq_len, hidden_size=128, lstm_layer=1):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.embedding = nn.Embedding(seq_len, hidden_size)  # Here probably use F.one_hot (yes, do that)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True) # add bidrectional flag
        self.out = nn.Linear(hidden_size, seq_len) # Second input should possibly be '2', because binary classification (might include bias term by default)

    def forward(self,input,hidden):
        # Or maybe just insert F.one_hot here?
        x = self.embedding(input).view(1,1,-1)
        x,hidden = self.lstm(x,hidden) # double check what lstms output
        output = self.out(x)
        return output,hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


# build forward fn
# build predict fn
# build eval fn
