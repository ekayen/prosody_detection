import numpy as np
import pickle
import torch
import random
import operator
from torch.nn.utils.rnn import pad_sequence

SEED = 123

def load_data(filename,shuffle=True,debug=True,max_len=None):
    data = []
    if '.txt' in filename:
        with open(filename,'r') as f:
            for line in f.readlines():
                tokens,labels = line.split('\t')
                tokens = [tok.strip() for tok in tokens.split()]
                labels = [int(i) for i in labels.split()]
                if max_len:
                    tokens = tokens[:max_len]
                    labels = labels[:max_len]
                labels = np.array(labels,dtype=np.int32)
                data.append((tokens,labels))
    elif '.pickle' in filename:
        with open(filename,'rb') as f:
            data = pickle.load(f)
    else:
        print("File format not recognized.")
    if shuffle:
        if debug:
            random.seed(SEED)
        random.shuffle(data)
    return data


def load_libri_data(filename,shuffle=True,debug=True,max_len=None):
    data = []
    if '.txt' in filename:
        with open(filename,'r') as f:
            for line in f.readlines():
                tokens,labels = line.split('\t')
                tokens = [tok.strip() for tok in tokens.split()]
                labels = [int(i) for i in labels.split()]
                if max_len:
                    tokens = tokens[:max_len]
                    labels = labels[:max_len]
                labels = np.array(labels,dtype=np.int32)
                data.append((tokens,labels))
    elif '.pickle' in filename:
        with open(filename,'rb') as f:
            data = pickle.load(f)
    else:
        print("File format not recognized.")
    if shuffle:
        if debug:
            random.seed(SEED)
        random.shuffle(data)
    return data

def to_ints(data,vocab_size,wd_to_i=None,i_to_wd=None): # TODO add UNK and PAD (figure out what the pytorch defaults for these are, if any)

    num_wds = []
    num_lbls = []

    if not wd_to_i:
        wd_to_i = {'<PAD>': 1,
                   '<UNK>': 0}
        i_to_wd = {1: '<PAD>',
                   0: '<UNK>'}
        wd_counts = {}
        for example in data:
            wds,lbls = example

            for wd in wds:
                if not wd in wd_counts:
                    wd_counts[wd] = 1
                else:
                    wd_counts[wd] += 1
        wds_by_freq = sorted(wd_counts.items(), key=operator.itemgetter(1), reverse=True)
        top_wds = [i[0] for i in wds_by_freq[:vocab_size]]
        counter = 2
        for wd in top_wds:
            wd_to_i[wd] = counter
            i_to_wd[counter] = wd
            counter += 1

    for example in data:
        wd_i = []
        wds,lbls = example
        for wd in wds:
            if wd in wd_to_i:
                wd_i.append(wd_to_i[wd])
            else:
                wd_i.append(wd_to_i['<UNK>'])
        num_wds.append(torch.tensor(wd_i,dtype=torch.long))
        num_lbls.append(torch.tensor(lbls,dtype=torch.long))

    return num_wds,num_lbls,wd_to_i,i_to_wd


class BatchWrapper:
    def __init__(self,in_iter,x,y):
        self.in_iter, self.x, self.y = in_iter,x,y

    def __iter__(self):
        for batch in self.in_iter:

            x = getattr(batch, self.x)
            y = getattr(batch, self.y)

            yield (x,y)

    def __len__(self):
        return len(self.in_iter)

def load_vectors(vector_file,wd_to_idx):
    vec_dict = {}
    with open(vector_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in wd_to_idx:
                wd_idx = wd_to_idx[word]
                vec_dict[wd_idx] = np.array(line[1:]).astype(np.float)
    return vec_dict

def shuffle_input_output(X,Y):
    data = list(zip(X,Y))
    random.shuffle(data)
    X,Y = zip(*data)
    return(X,Y)

def make_batches(X,Y,batch_size,device):
    X,Y = shuffle_input_output(X,Y)
    counter = 0
    start = 0
    end = batch_size
    batched_X = []
    batched_Y = []
    while end < len(X):
        X0 = X[start:end]
        Y0 = Y[start:end]
        X0 = pad_sequence(X0).to(device)
        Y0 = pad_sequence(Y0).to(device)
        batched_X.append(X0)
        batched_Y.append(Y0)
        start = end
        end = end + batch_size
    return(batched_X,batched_Y)
