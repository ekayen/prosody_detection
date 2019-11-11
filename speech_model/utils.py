import torch
from torch.utils import data
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import pickle
import pandas as pd


random.seed(0)
torch.manual_seed(123)

class SynthDataset(data.Dataset):
    def __init__(self, list_ids, utt_dict,label_dict):
        self.list_ids = list_ids
        self.utt_dict = utt_dict
        self.label_dict = label_dict

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        id = self.list_ids[index]
        X = self.utt_dict[id]
        y = self.label_dict[id]
        return X, y

class UttDataset(data.Dataset):
    def __init__(self, list_ids, utt_dict,label_dict,pad_len):
        self.list_ids = list_ids
        self.utt_dict = utt_dict
        self.label_dict = label_dict
        self.pad_len = pad_len

    def pad_left(self,arr):
        if arr.shape[0] < self.pad_len:
            dff = self.pad_len - arr.shape[0]
            arr = F.pad(arr,pad=(0,0,dff,0),mode='constant')
        else:
            arr = arr[arr.shape[0]-self.pad_len:]
        return arr

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        id = self.list_ids[index]
        X = self.pad_left(self.utt_dict[id])
        #X = self.utt_dict[id]
        y = self.label_dict[id]
        return X, y

class UttDatasetWithToktimes(UttDataset):
    def __init__(self,list_ids,utt_dict,label_dict,toktimes_dict,pad_len):
        super(UttDatasetWithToktimes,self).__init__(list_ids, utt_dict,label_dict,pad_len)
        self.toktimes_dict = toktimes_dict

    def pad_right(self,arr):
        if arr.shape[0] < self.pad_len:
            dff = self.pad_len - arr.shape[0]
            arr = F.pad(arr,pad=(0,0,0,dff),mode='constant')
        else:
            arr = arr[:self.pad_len]
        return arr

    def __getitem__(self, index):
        id = self.list_ids[index]
        toks = self.pad_right(self.utt_dict[id])
        toktimes = self.toktimes_dict[id]
        X = (toks,toktimes)
        y = self.label_dict[id]
        return X,y


class UttDatasetWithId(UttDataset):
    def __getitem__(self, index):
        id = self.list_ids[index]
        X = self.pad_left(self.utt_dict[id])
        # X = self.utt_dict[id]
        y = self.label_dict[id]
        return id, X, y


def plot_grad_flow(named_parameters):
    '''
    From user RoshanRone here:
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/6
    '''
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def plot_results(train_losses, train_accs, dev_accs, train_steps,model_name):
    df = pd.DataFrame(dict(train_steps=train_steps,
                           train_losses=train_losses,
                           train_accs=train_accs,
                           dev_accs=dev_accs))

    with open("tmp.pkl", 'wb') as f:
        pickle.dump(df, f)

    ax = plt.gca()
    df.plot(kind='line', x='train_steps', y='train_losses', ax=ax)
    df.plot(kind='line', x='train_steps', y='train_accs', color='red', ax=ax)
    df.plot(kind='line', x='train_steps', y='dev_accs', color='green', ax=ax)

    plt.savefig('results/{}.png'.format(model_name))
    plt.show()
    df.to_csv('results/{}.tsv'.format(model_name), sep='\t')

