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

def print_progress(progress, info='', bar_len=20):
	filled = int(progress*bar_len)
	print('\r[{}{}] {:.2f}% {}'.format('=' * filled, ' ' * (bar_len-filled), progress*100, info), end='')

class BurncDataset(data.Dataset):
    def __init__(self,cfg,input_dict,mode='train',datasplit=None):

        self.segmentation = cfg['segmentation']
        self.context_window = cfg['context_window']
        self.feats = cfg['feats']
        self.bitmark = cfg['bitmark']

        self.pad_len = cfg['pad_len']
        self.input_dict = input_dict
        self.mode = mode

        if not datasplit:
            datasplit = cfg['datasplit']

        # From the config file, load the ids you need
        with open(datasplit, 'r') as f:
            split_ids = yaml.load(f, yaml.FullLoader)
        self.utt_ids = split_ids[self.mode]
        if self.segmentation=='tokens':
            self.ids = [tok for utt_id in self.utt_ids for tok in input_dict['utt2toks'][utt_id]]
        elif self.segmentation=='utterances':
            self.ids = self.utt_ids

    def __len__(self):
        return len(self.ids)


class BurncDatasetSpeech(BurncDataset):
    def __init__(self, config, input_dict, mode='train',datasplit=None):
        super(BurncDatasetSpeech,self).__init__(config, input_dict, mode, datasplit)


    def pad_right(self,arr):
        if arr.shape[0] < self.pad_len:
            dff = self.pad_len - arr.shape[0]
            arr = F.pad(arr,pad=(0,0,0,dff),mode='constant')
        else:
            arr = arr[:self.pad_len]
        return arr

    def __getitem__(self, index):
        id = self.ids[index]
        #para_id = id.split('-')[0]
        if self.segmentation == 'tokens':
            tok_ids = [id]
            if self.context_window:
                curr_utt = self.input_dict['tok2utt'][id]
                prev_idx = self.input_dict['utt2toks'][curr_utt].index(id) - 1
                next_idx = self.input_dict['utt2toks'][curr_utt].index(id) + 1
                if prev_idx >= 0:
                    prev_id = self.input_dict['utt2toks'][curr_utt][prev_idx]
                    tok_ids = [prev_id] + tok_ids
                if next_idx < len(self.input_dict['utt2toks'][curr_utt]):
                    next_id = self.input_dict['utt2toks'][curr_utt][next_idx]
                    tok_ids.append(next_id)
            tok_feats = [self.input_dict[self.feats][i] for i in tok_ids]
            if self.context_window and self.bitmark: # TODO this is redundant
                tmp = []
                for i,feats in enumerate(tok_feats):
                    if i==1:
                        ones = torch.ones((feats.shape[0],1))
                        feats = torch.cat((feats,ones),dim=1)
                        tmp.append(feats)
                    else:
                        zeros = torch.zeros((feats.shape[0],1))

                        feats = torch.cat((feats,zeros),dim=1)
                        tmp.append(feats)
                tok_feats = tmp
            X = torch.cat(tok_feats,dim=0)
            #X = self.pad_right(X)
            Y = torch.tensor(self.input_dict['tok2tone'][id])
            toktimes = [self.input_dict['tok2times'][i][0] for i in tok_ids] + [self.input_dict['tok2times'][tok_ids[-1]][1]]

            if self.context_window and len(toktimes)< 4:
                while len(toktimes) < 4:
                    toktimes.append(toktimes[-1])
            #if not len(toktimes)==4: print(id,len(toktimes))
            initial_time = toktimes[0]
            toktimes = torch.tensor([int(round(100*(tim-initial_time))) for tim in toktimes],dtype=torch.float32)


        elif self.segmentation == 'utterances':
            tok_ids = self.input_dict['utt2toks'][id]
            X = torch.cat([self.input_dict[self.feats][tok_id] for tok_id in tok_ids], dim=0)
            #X = self.pad_right(X)
            Y = torch.tensor([self.input_dict['tok2tone'][tok_id] for tok_id in tok_ids])

            toktimes = self.input_dict['utt2frames'][id]
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

def plot_results(plot_data,model_name,results_path):
    df = pd.DataFrame(dict(epochs=plot_data['time'],
                           train_losses=plot_data['loss'],
                           train_accs=plot_data['train_acc'],
                           dev_accs=plot_data['dev_acc']))

    with open("tmp.pkl", 'wb') as f:
        pickle.dump(df, f)

    ax = plt.gca()
    df.plot(kind='line', x='epochs', y='train_losses', ax=ax)
    df.plot(kind='line', x='epochs', y='train_accs', color='red', ax=ax)
    df.plot(kind='line', x='epochs', y='dev_accs', color='green', ax=ax)

    plt.savefig('{}/{}.png'.format(results_path,model_name))
    plt.show()
    df.to_csv('{}/{}.tsv'.format(results_path,model_name), sep='\t')

def gen_model_name(cfg):
    """
    name_sections = []
    name_sections.append(['model_name'])
    name_sections.append(f's{cfg["seed"]}')
    name_sections.append(f'cnn{cfg["cnn_layers"]}')
    name_sections.append(f'lstm{cfg["lstm_layers"]}')
    name_sections.append(f'd{cfg["dropout"]}')
    return '_'.join(name_sections)
    """
    return cfg['model_name']

def main():
    # FOR TESTING ONLY
    cfg_file = 'conf/replication_pros.yaml'
    with open(cfg_file, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    burnc_dict = '../data/burnc/burnc.pkl'
    with open(burnc_dict,'rb') as f:
        input_dict = pickle.load(f)
    dataset = BurncDatasetSpeech(cfg, input_dict, mode='dev')
    datagen = data.DataLoader(dataset, **cfg['train_params'])
    lens = []
    for id,(x,toktimes),y in datagen:
        lens.append(x.shape[1])
    print(max(lens))
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()