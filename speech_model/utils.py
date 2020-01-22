import torch
from torch.utils import data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import yaml
from tabulate import tabulate
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from decimal import Decimal

class BurncDataset(data.Dataset):
    def __init__(self,cfg,input_dict, w2i, vocab_size=3000,mode='train',datasplit=None,overwrite_speech=False,
                 scramble_speech=False,stopwords_only=False):

        self.segmentation = cfg['segmentation']
        self.context_window = cfg['context_window']
        self.feats = cfg['feats']
        self.bitmark = cfg['bitmark']

        self.frame_pad_len = cfg['frame_pad_len']
        self.tok_pad_len = cfg['tok_pad_len']
        self.input_dict = input_dict
        self.mode = mode

        self.vocab_size = vocab_size
        self.w2i = self.adjust_vocab_size(w2i)

        self.overwrite_speech = overwrite_speech
        self.scramble_speech = scramble_speech
        self.stopwords_only = stopwords_only

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

    def pad_right(self,arr,pad_len,num_dims=2):
        if arr.shape[0] < pad_len:
            dff = pad_len - arr.shape[0]
            if num_dims==2: # For padding 2d speech data
                arr = F.pad(arr, pad=(0, 0, 0, dff), mode='constant')
            elif num_dims==1: # For padding 1d string data
                arr = F.pad(arr, pad=(0, dff), mode='constant')
        else:
            arr = arr[:pad_len]
        return arr

    def get_context(self,tok_ids):
        iden = tok_ids[0]
        curr_utt = self.input_dict['tok2utt'][iden]
        prev_idx = self.input_dict['utt2toks'][curr_utt].index(iden) - 1
        next_idx = self.input_dict['utt2toks'][curr_utt].index(iden) + 1
        if prev_idx >= 0:
            prev_id = self.input_dict['utt2toks'][curr_utt][prev_idx]
            tok_ids = [prev_id] + tok_ids
        if next_idx < len(self.input_dict['utt2toks'][curr_utt]):
            next_id = self.input_dict['utt2toks'][curr_utt][next_idx]
            tok_ids.append(next_id)
        return tok_ids

    def get_toktimes(self,iden,tok_ids):
        if self.segmentation == 'utterances':
            toktimes = self.input_dict['utt2frames'][iden]
            toktimes = self.pad_right(toktimes, self.tok_pad_len + 1, num_dims=1)
        if self.segmentation == 'tokens':
            toktimes = [self.input_dict['tok2times'][i][0] for i in tok_ids] + [
                self.input_dict['tok2times'][tok_ids[-1]][1]]

            if self.context_window and len(toktimes) < 4:
                while len(toktimes) < 4:
                    toktimes.append(toktimes[-1])
            initial_time = toktimes[0]
            toktimes = torch.tensor([int(round(100 * (tim - initial_time))) for tim in toktimes], dtype=torch.float32)
        return toktimes

    def get_tok_ids(self,iden):
        if self.segmentation=='tokens':
            tok_ids = [iden]
            if self.context_window:
                tok_ids = self.get_context(tok_ids)
        if self.segmentation=='utterances':
            tok_ids = self.input_dict['utt2toks'][iden]
        return tok_ids

    def get_speech_feats(self,tok_ids):
        tok_feats = [self.input_dict[self.feats][tok_id] for tok_id in tok_ids]
        if self.bitmark and self.segmentation=='tokens':
            tmp = []
            for i, feats in enumerate(tok_feats):
                if i == 1:
                    ones = torch.ones((feats.shape[0], 1))
                    feats = torch.cat((feats, ones), dim=1)
                    tmp.append(feats)
                else:
                    zeros = torch.zeros((feats.shape[0], 1))

                    feats = torch.cat((feats, zeros), dim=1)
                    tmp.append(feats)
            tok_feats = tmp
        X = torch.cat(tok_feats, dim=0)
        if self.overwrite_speech:
            X = torch.ones(X.shape)
        if self.scramble_speech:
            X = X[torch.randperm(X.size()[0])]
        if self.frame_pad_len:
            X = self.pad_right(X, self.frame_pad_len)
        return X

    def get_labels(self,tok_ids,iden=None):
        if self.segmentation=='tokens':
            tok_ids = [iden]
        Y = torch.tensor([self.input_dict['tok2tone'][tok_id] for tok_id in tok_ids])

        if self.tok_pad_len and not self.segmentation=='tokens':
            Y = self.pad_right(Y, self.tok_pad_len, num_dims=1)
        if self.segmentation=='tokens':
            Y = Y.squeeze()
        return Y

    def adjust_vocab_size(self,w2i):
        for wd in w2i:
            if w2i[wd] > self.vocab_size:
                w2i[wd] = w2i['UNK']
        return w2i

    def uniformize_vocab(self,tok_ints):
        '''
        Helper fn for doing experiments where the model only gets a 'stopword or not' signal
        This function takes a token seq and replaces all non-pad, non-UNK tokens with 2
        '''
        STOPWD_IDX = 2
        return [i if i in (self.w2i['PAD'],self.w2i['UNK']) else STOPWD_IDX for i in tok_ints]



    def get_tokens(self,tok_ids):
        tok_ints = []
        for tok_id in tok_ids:
            if self.input_dict['tok2str'][tok_id] in self.w2i:
                tok_ints.append(self.w2i[self.input_dict['tok2str'][tok_id]])
            else:
                tok_ints.append(self.w2i['UNK'])
        if self.stopwords_only:
            tok_ints  = self.uniformize_vocab(tok_ints)
        tok_ints = torch.tensor(tok_ints)
        if self.tok_pad_len:
            tok_ints = self.pad_right(tok_ints, self.tok_pad_len, num_dims=1)
        return tok_ints

    def __getitem__(self, index):
        iden = self.ids[index]
        tok_ids = self.get_tok_ids(iden)
        labels = self.get_labels(tok_ids,iden)
        toktimes = self.get_toktimes(iden,tok_ids)
        tok_ints = self.get_tokens(tok_ids)
        speech_feats = self.get_speech_feats(tok_ids)

        return iden, (speech_feats,tok_ints,toktimes), labels

    
class BurncDatasetSpeech(BurncDataset):
    def __init__(self, config, input_dict, w2i, vocab_size=3000, mode='train',datasplit=None):
        super(BurncDatasetSpeech,self).__init__(config, input_dict, w2i, vocab_size, mode, datasplit)

    def __getitem__(self, index):

        iden = self.ids[index]
        tok_ids = self.get_tok_ids(iden)
        speech_feats = self.get_speech_feats(tok_ids)
        toktimes = self.get_toktimes(iden,tok_ids)
        labels = self.get_labels(tok_ids, iden)

        return iden, (speech_feats, toktimes), labels

class BurncDatasetText(BurncDataset):

    def __init__(self, config, input_dict, w2i, vocab_size=3000, mode='train',datasplit=None):
        super(BurncDatasetText,self).__init__(config, input_dict, w2i, vocab_size, mode, datasplit)
        self.pad_len = config['tok_pad_len']

    def __getitem__(self, index):
        iden = self.ids[index]
        tok_ids = self.get_tok_ids(iden)
        labels = self.get_labels(tok_ids,iden)
        toktimes = self.get_toktimes(iden,tok_ids)
        tok_ints = self.get_tokens(tok_ids)

        return iden, (tok_ints,toktimes), labels

class SynthDataset(data.Dataset):
    def __init__(self, list_ids, utt_dict,label_dict):
        self.list_ids = list_ids
        self.utt_dict = utt_dict
        self.label_dict = label_dict

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        iden = self.list_ids[index]
        X = self.utt_dict[iden]
        y = self.label_dict[iden]
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
        iden = self.list_ids[index]
        X = self.pad_left(self.utt_dict[iden])
        #X = self.utt_dict[iden]
        y = self.label_dict[iden]
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
        iden = self.list_ids[index]
        toks = self.pad_right(self.utt_dict[iden])
        toktimes = self.toktimes_dict[iden]
        X = (toks,toktimes)
        y = self.label_dict[iden]
        return iden,X,y

class UttDatasetWithId(UttDataset):
    def __getitem__(self, index):
        iden = self.list_ids[index]
        X = self.pad_left(self.utt_dict[iden])
        # X = self.utt_dict[iden]
        y = self.label_dict[iden]
        return iden, X, y

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

def gen_model_name(cfg,datasplit):
    name_sections = []
    name_sections.append(cfg['model_name'])
    datasplit = datasplit.split('/')[-1].split('.')[0]
    name_sections.append(datasplit)
    name_sections.append(f's{cfg["seed"]}')
    name_sections.append(f'cnn{cfg["cnn_layers"]}')
    if cfg['include_lstm']:
        name_sections.append(f'lstm{cfg["lstm_layers"]}')
    dropout = int(float(cfg['dropout'])*10)
    name_sections.append(f'd{dropout}')
    lr = '{:.0e}'.format(Decimal(cfg['learning_rate']))
    name_sections.append(f'lr{lr}')
    if not cfg['weight_decay']==0:
        #wd = int(cfg['weight_decay']*100000)
        wd = '{:.0e}'.format(Decimal(cfg['weight_decay']))
        name_sections.append(f'wd{wd}')
    name_sections.append(f'f{cfg["frame_filter_size"]}')
    name_sections.append(f'p{cfg["frame_pad_size"]}')
    name_sections.append(f'{cfg["flatten_method"]}')
    name_sections.append(f'b{cfg["bottleneck_feats"]}')
    name_sections.append(f'h{cfg["hidden_size"]}')
    name_sections.append(f'e{cfg["embedding_dim"]}')
    name_sections.append(f'v{cfg["vocab_size"]}')
    return '_'.join(name_sections)

def report_hparams(cfg,datasplit=None):

    if not datasplit: datasplit = cfg['datasplit']

    to_print = [
                    ['Model name', gen_model_name(cfg,datasplit)],
                    ['Datasplit', datasplit],
                    ['Features', cfg['feats']],
                    ['Pad_len', cfg['pad_len']],
                    ['Feature dim', cfg['feat_dim']],
                    ['CNN layers', cfg['cnn_layers']],
                    ['Dropout', cfg['dropout']],
                    ['Learning rate', cfg['learning_rate']],
                    ['Epochs', cfg['num_epochs']],
                    ['Seed', cfg['seed']]
                ]
    if cfg['include_lstm']:
        lstm_params = [['LSTM layers',cfg['lstm_layers']],
                       ['LSTM hidden size',cfg['hidden_size']],
                       ['Bidirectional',cfg['bidirectional']]]
        to_print = to_print + lstm_params

    print()
    print(tabulate(to_print,
                   headers=['Hparam','Value']))
    print()

def set_seeds(seed):
    print(f'setting seed to {seed}')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_progress(progress, info='', bar_len=20):
	filled = int(progress*bar_len)
	print('\r[{}{}] {:.2f}% {}'.format('=' * filled, ' ' * (bar_len-filled), progress*100, info), end='')

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

def load_vectors(vector_file,wd_to_idx):
    '''
    Load pre-trained embeddings as dict
    '''
    vec_dict = {}
    with open(vector_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in wd_to_idx:
                wd_idx = wd_to_idx[word]
                vec_dict[wd_idx] = np.array(line[1:]).astype(np.float)
    return vec_dict


def main():
    # FOR TESTING ONLY
    #cfg_file = 'conf/replication_pros.yaml'
    cfg_file = 'conf/cnn_lstm_pros.yaml'
    with open(cfg_file, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    burnc_dict = '../data/burnc/burnc.pkl'
    with open(burnc_dict,'rb') as f:
        input_dict = pickle.load(f)
    splt = cfg['datasplit'].split('/')[-1].split('.')[0]
    vocab_file = f'../data/burnc/splits/{splt}.vocab'
    print(vocab_file)
    with open(vocab_file, 'rb') as f:
        vocab_dict = pickle.load(f)

    dataset = BurncDataset(cfg,input_dict, vocab_dict['w2i'], vocab_size=30,mode='train',datasplit=None,overwrite_speech=False,scramble_speech=True,stopwords_only=True)
    item = dataset.__getitem__(4)

    import pdb;pdb.set_trace()
    dataset = BurncDatasetSpeech(cfg, input_dict, mode='dev')
    item = dataset.__getitem__(4)

    import pdb;pdb.set_trace()
    dataset = BurncDatasetText(cfg, input_dict, vocab_dict['w2i'], vocab_size=3000, mode='train',datasplit=None)
    item = dataset.__getitem__(4)
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    main()
