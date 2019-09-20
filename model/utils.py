import numpy as np
import pickle
import torch

def load_data(filename):
    data = []
    if '.txt' in filename:
        with open(filename,'r') as f:
            for line in f.readlines():
                tokens,labels = line.split('\t')
                tokens = [tok.strip() for tok in tokens.split()]
                labels = np.array([int(i) for i in labels.split()],dtype=np.int32)
                data.append((tokens,labels))
    elif '.pickle' in filename:
        with open(filename,'rb') as f:
            data = pickle.load(f)
    else:
        print("File format not recognized.")
    return data

def to_ints(data):
    wd_to_i = {}
    i_to_wd = {}
    counter = 0
    num_wds = []
    num_lbls = []
    for example in data:
        wds,lbls = example
        wd_i = []
        for wd in wds:
            if not wd in wd_to_i:
                wd_to_i[wd] = counter
                i_to_wd[counter] = wd
                counter += 1
            wd_i.append(wd_to_i[wd])
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
