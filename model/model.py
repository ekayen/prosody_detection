# Get words in one-hot format
# Build model
from torch import nn
from torchtext import data,datasets
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# Use torchtext.vocab to numericalize the text and pad the seqs


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


"""
class BiLSTM(nn.module):
    def __init__(self, padding_idx, static=True, hidden_size=128, lstm_layer=2, dropout=0.2):
        super(BiLSTM,self).__init__()
        self.hidden_size = hidden_size
#        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self):


"""

# build forward fn
# build predict fn
# build eval fn
