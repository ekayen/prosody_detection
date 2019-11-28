import torch
from torch.utils import data
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import yaml

random.seed(0)
torch.manual_seed(123)

class BurncDataset(data.Dataset):
    def __init__(self,config,input_dict,mode='train'):

        # Load the config file for the whole model
        with open(config, 'r') as f:
            cfg = yaml.load(f, yaml.FullLoader)
        self.segmentation = cfg['segmentation']
        self.context_window = cfg['context_window']
        self.feats = cfg['feats']
        self.bitmark = cfg['bitmark']

        self.input_dict = input_dict
        self.mode = mode

        # From the config file, load the ids you need
        with open(cfg['datasplit'], 'r') as f:
            split_ids = yaml.load(f, yaml.FullLoader)
        self.utt_ids = split_ids[self.mode]
        if self.segmentation=='tokens':
            self.ids = [tok for utt_id in self.utt_ids for tok in input_dict[utt_id.split('-')[0]]['utterances'][utt_id] ]
        elif self.segmentation=='utterances':
            self.ids = self.utt_ids

    def __len__(self):
        return len(self.ids)


class BurncDatasetSpeech(BurncDataset):
    def __init__(self, config, input_dict, mode='train'):
        super(BurncDatasetSpeech,self).__init__(config, input_dict, mode)

    def __getitem__(self, index):
        id = self.ids[index]
        para_id = id.split('-')[0]
        if self.segmentation == 'tokens':
            tok_ids = [id]
            if self.context_window:
                curr_utt = self.input_dict[para_id]['tok2utt'][id]
                prev_idx = self.input_dict[para_id]['utterances'][curr_utt].index(id) - 1
                next_idx = self.input_dict[para_id]['utterances'][curr_utt].index(id) + 1
                if prev_idx >= 0:
                    prev_id = self.input_dict[para_id]['utterances'][curr_utt][prev_idx]
                    tok_ids = [prev_id] + tok_ids
                if next_idx <= len(self.input_dict[para_id]['utterances'][curr_utt]):
                    next_id = self.input_dict[para_id]['utterances'][curr_utt][next_idx]
                    tok_ids.append(next_id)
            tok_feats = [self.input_dict[para_id][self.feats][i] for i in tok_ids]
            X = torch.cat(tok_feats,dim=0)
            Y = torch.tensor(self.input_dict[para_id]['tok2tone'][id])
            toktimes = [self.input_dict[para_id]['tok2times'][i][0] for i in tok_ids] + [self.input_dict[para_id]['tok2times'][tok_ids[-1]][1]]

        elif self.segmentation == 'utterances':
            tok_ids = self.input_dict[para_id]['utterances'][id]
            X = torch.cat([self.input_dict[para_id][self.feats][tok_id] for tok_id in tok_ids], dim=0)
            Y = torch.tensor([self.input_dict[para_id]['tok2tone'][tok_id] for tok_id in tok_ids])
            toktimes = [self.input_dict[para_id]['tok2times'][tok_id][0] for tok_id in tok_ids] + \
                       [self.input_dict[para_id]['tok2times'][tok_ids[-1]][1]]
        import pdb;pdb.set_trace()
        return id, (X, toktimes), Y

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
        return id,X,y

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

def plot_results(train_losses, train_accs, dev_accs, train_steps,model_name,results_path):
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

    plt.savefig('{}/{}.png'.format(results_path,model_name))
    plt.show()
    df.to_csv('{}/{}.tsv'.format(results_path,model_name), sep='\t')


import pickle
with open('../data/burnc/burnc.pkl','rb') as f:
    input_dict = pickle.load(f)

dataset = BurncDatasetSpeech(config='tmp_config.yaml',input_dict=input_dict,mode='train')

dataset.__getitem__(4)
