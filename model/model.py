# Get words in one-hot format
# Build model
from torch import nn
import torch
torch.manual_seed(0)

from torchtext import data,datasets
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils import load_data,BatchWrapper,to_ints
import numpy as np
from seqeval.metrics import accuracy_score, classification_report,f1_score

PRINT_DIMS = False
PRINT_EVERY = 50
EVAL_EVERY = 50

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 3
BIDIRECTIONAL = True
LEARNING_RATE = 0.0001
TRAIN_RATIO = 0.6
DEV_RATIO = 0.2
VOCAB_SIZE = 4000
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


def make_batches(X,Y,batch_size):
    counter = 0
    start = 0
    end = batch_size
    batched_X = []
    batched_Y = []
    """
    X0 = X[start:end]
    Y0 = Y[start:end]
    X0 = pad_sequence(X0)
    Y0 = pad_sequence(Y0)
    batched_X.append(X0)
    batched_Y.append(Y0)
    """
    while end < len(X):
        X0 = X[start:end]
        Y0 = Y[start:end]
        X0 = pad_sequence(X0)
        Y0 = pad_sequence(Y0)
        batched_X.append(X0)
        batched_Y.append(Y0)
        start = end
        end = end + batch_size
    return(batched_X,batched_Y)



X_train_batches,Y_train_batches = make_batches(X_train,Y_train,BATCH_SIZE)
X_dev_batches,Y_dev_batches = make_batches(X_dev,Y_dev,BATCH_SIZE)
X_test_batches,Y_test_batches = make_batches(X_test,Y_test,BATCH_SIZE)



"""
for i,instance in enumerate(X_train):
    if instance.sum().item() == 0:
        print(i, instance)
        import pdb;pdb.set_trace()
"""

# BUILD THE MODEL

class BiLSTM(nn.Module):
    # LSTM: (embedding_dim, hidden_size)
    # OUTPUT = (hidden size, tagset_size),
    # SOFTMAX over dimension 2 (I am not sure if this is right)
    def __init__(self, batch_size, vocab_size, tagset_size, embedding_dim=100, hidden_size=128, lstm_layers=1, bidirectional=True, output_dim=1):

        super(BiLSTM,self).__init__()

        # hparams:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.output_dim = output_dim

        # layers:
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=self.bidirectional, num_layers=lstm_layers)
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

    def init_hidden(self):

        if self.bidirectional:
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.lstm_layers*2, self.batch_size, self.hidden_size).requires_grad_()
            c0 = torch.zeros(self.lstm_layers*2, self.batch_size, self.hidden_size).requires_grad_()
        else:
            h0 = torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size).requires_grad_()
            c0 = torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size).requires_grad_()

        return (h0,c0)

# INSTANTIATE THE MODEL

model = BiLSTM(batch_size=BATCH_SIZE,vocab_size=VOCAB_SIZE+2,tagset_size=2,bidirectional=BIDIRECTIONAL)
#loss_fn = nn.CrossEntropyLoss() # TODO change to binary crossentropy loss (maybe)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

"""
def find_end_idx(input,pad_char=0):
    idx = []
    input = torch.transpose(input,0,1).numpy().tolist()
    for seq in input:
        if pad_char in seq:
            idx.append(seq.index(pad_char))
        else:
            idx.append(len(input))
    return idx

def unpad_seqs(input,output):
    indices = find_end_idx(input)
    tmp = []
    for i,out in enumerate(output):
        tmp.append(out[:indices[i]])
    return tmp
"""
# DEFINE EVAL FUNCTION

def evaluate(X, Y,mdl):
    print("EVAL=================================================================================================")
    y_pred = []
    with torch.no_grad():
        for i in range(len(X)):
            #input = torch.tensor([i if i in i_to_wd else wd_to_i['<UNK>'] for i in input])
            input = X[i]

            if not (list(input.shape)[0] == 0):
                hidden = mdl.init_hidden()
                tag_scores, _ = mdl(input, hidden)
                #pred = np.squeeze(np.argmax(tag_scores, axis=-1)).tolist() # TODO could this be wrong? Almost certainly yes.

                #pred = torch.transpose(tag_scores,0,1)
                pred = np.where(tag_scores>0.5,1,0)
                pred = np.squeeze(pred)
                #pred = pred.tolist()
                if type(pred) is int:
                    pred = [pred]
                #pred = [[str(j) for j in i] for i in pred]

                #y_pred += pred
                y_pred.append(pred)
    print('Evaluation:')
    Y = np.concatenate([y.flatten() for y in Y]).tolist()
    y_pred = np.concatenate([y.flatten() for y in y_pred]).tolist()
    Y = [str(y) for y in Y]
    y_pred = [str(y) for y in y_pred]

    print('F1:',f1_score(Y, y_pred))
    print('Acc',accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

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

# TRAIN
recent_losses = []
for epoch in range(NUM_EPOCHS):
    #for i in range(len(X_train)):
    print("TRAIN================================================================================================")
    for i in range(len(X_train_batches)):

        #input,labels = X_train[i],Y_train[i]
        input, labels = X_train_batches[i], Y_train_batches[i]

        if not (list(input.shape)[0] == 0):

            model.zero_grad()

            hidden = model.init_hidden()
            tag_scores,_ = model(input,hidden)

            #import pdb;  pdb.set_trace()
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
                #evaluate(X_dev,Y_dev_str,model)
                evaluate(X_dev_batches, Y_dev_batches, model)


"""
print('After training, train:')
evaluate(X_train,Y_train_str,model)


print('After training, dev: ')
evaluate(X_dev, Y_dev_str,model)

"""







