import argparse
from utils import SwbdDatasetInfostruc
import yaml
import pickle
from torch.utils import data

parser = argparse.ArgumentParser()
parser.add_argument("-c","--config", help="path to config file", default='conf/swbd_analysis.yaml')

args = parser.parse_args()
with open(args.config, 'r') as f:
    cfg = yaml.load(f, yaml.FullLoader)
datasplit = cfg['datasplit']

with open(cfg['all_data'], 'rb') as f:
    data_dict = pickle.load(f)

with open(cfg['datasplit'].replace('yaml', 'vocab'), 'rb') as f:
    vocab_dict = pickle.load(f)

w2i = vocab_dict['w2i']

trainset = SwbdDatasetInfostruc(cfg, data_dict, w2i, cfg['vocab_size'], mode='train', datasplit=datasplit,
                                labelling=cfg['labelling'],vocab_dict=vocab_dict)
devset = SwbdDatasetInfostruc(cfg, data_dict, w2i, cfg['vocab_size'], mode='dev', datasplit=datasplit,
                              labelling=cfg['labelling'],vocab_dict=vocab_dict)

traingen = data.DataLoader(trainset, **cfg['train_params'])
devgen = data.DataLoader(devset, **cfg['train_params'])

tok2tone = data_dict['tok2tone']
utt2toks = data_dict['utt2toks']

new_acc = 0
new_nonacc = 0
nonnew_acc = 0
nonnew_nonacc = 0
for id, (speech,text,toktimes), labels in traingen:
    tones = [tok2tone[tok] for tok in utt2toks[id[0]]]
    labels = labels.flatten().tolist()[:len(tones)]
    if sum(labels) > 0:
        for i,new in enumerate(labels):
            if new == 1:
                if tones[i] == 1:
                    new_acc += 1
                else:
                    new_nonacc += 1
            else:
                if tones[i] == 1:
                    nonnew_acc += 1
                else:
                    nonnew_nonacc += 1

print(f'\tNew\tNonnew')
print(f'Acc\t{new_acc}\t{nonnew_acc}')
print(f'Nonacc\t{new_nonacc}\t{nonnew_nonacc}')

