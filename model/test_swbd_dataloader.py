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
import os
import nltk
import time
from swbd_dataloader import SwbdSegDataset
from utils import SwbdDatasetInfostruc
from swbd_vocab_dict import create_vocab_dict

"""
"""

#model = SpeechEncoder()
with open('conf/swbd.yaml', 'r') as f:
    cfg = yaml.load(f, yaml.FullLoader)

    
mode = 'train'
"""
vocab_dict_path = os.path.join(cfg['speech_dir'],'seg','vocab_dict.pickle')


if os.path.exists(vocab_dict_path):
    vocab_dict = pickle.load(open(vocab_dict_path,'rb'))
    w2i = vocab_dict['w2i']
    i2w = vocab_dict['i2w']
else:

if True:
    w2i,i2w = create_vocab_dict(os.path.join(cfg['speech_dir'],'seg',f'turn_{mode}.txt'))
    with open(vocab_dict_path,'wb') as f:
        vocab_dict = {'w2i':w2i,
                      'i2w':i2w}
        pickle.dump(vocab_dict,f)
"""
vocab_dict_path = cfg['datasplit'].replace('yaml','vocab')
vocab_dict = pickle.load(open(vocab_dict_path,'rb'))
w2i = vocab_dict['w2i']


input_dict = pickle.load(open(cfg['all_data'],'rb'))

#trainset = SwbdSegDataset(cfg, w2i, mode='train')
trainset = SwbdDatasetInfostruc(cfg,input_dict, w2i, mode='train')
traingen = data.DataLoader(trainset, **cfg['train_params'])


max_frames = 0
max_toks = 0
for idx in range(len(trainset)):
    iden, frames, pause,dur, labels, toktimes, tok_ints = trainset.__getitem__(idx)
    import pdb;pdb.set_trace()
    max_frames = max(frames.shape[-1],max_frames)
    max_toks = max(tok_ints.shape[0],max_toks)
    

"""
for example in traingen:

    # The speech needs to be a single matrix of size feats x frames
    # The text ???
    # The toktimes needs to be a list of token times
    # get hidden using model.init_hidden()
    prediction = model(speech,text,toktimes,hidden)
"""
