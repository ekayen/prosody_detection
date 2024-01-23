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
import nltk
import time
import os
from nltk.corpus import stopwords,cmudict

class SwbdSegDataset(data.Dataset):
    def __init__(self,cfg,w2i,mode='train'):

        self.vocab_size = cfg['vocab_size']
        self.w2i = self.adjust_vocab_size(w2i)

        
        self.speech_dir = cfg['speech_dir']
        self.partition = pickle.load(open(os.path.join(self.speech_dir,f'turn_{mode}_partition.pickle'),'rb'))
        self.pitch = pickle.load(open(os.path.join(self.speech_dir,f'turn_{mode}_pitch.pickle'),'rb'))
        self.fbank = pickle.load(open(os.path.join(self.speech_dir,f'turn_{mode}_fbank.pickle'),'rb'))
        self.dur = pickle.load(open(os.path.join(self.speech_dir,f'turn_{mode}_duration.pickle'),'rb'))
        self.pause = pickle.load(open(os.path.join(self.speech_dir,f'turn_{mode}_pause.pickle'),'rb'))

        self.label_file = os.path.join(self.speech_dir,'seg',f'turn_{mode}.lbl')
        self.text_file = os.path.join(self.speech_dir,'seg',f'turn_{mode}.txt') 
        self.id_file = os.path.join(self.speech_dir,'seg',f'turn_{mode}.ids')
        
        self.mode = mode

        self.ids = self.load_ids()
        self.lbls = self.load_lbls()
        self.text = self.load_text()
        self.tokids = self.text2tokid()
        self.id2lbl = dict(zip(self.ids,self.lbls))
        self.id2text = dict(zip(self.ids,self.text))
        self.id2tokids = dict(zip(self.ids,self.tokids))
        

        self.frame_pad_len = cfg['frame_pad_len']
        self.tok_pad_len = cfg['tok_pad_len']
            

    def __len__(self):
        return len(self.ids)

    def get_frames(self,iden):
        parts = self.partition[iden]
        utt_pitch = self.pitch[iden]
        utt_fbank = self.fbank[iden]
        utt_frames = np.concatenate((utt_pitch,utt_fbank),axis=0)
        # TODO make it possible to ablate one of these feats
        """
        for idx in range(len(parts)):
            if idx+1 < len(parts):
                if parts[idx][-1] < parts[idx+1][0]:
                    utt_frames[:,parts[idx][-1]+1:parts[idx+1][0]] = 0
        """
        utt_frames = torch.tensor(utt_frames,dtype=torch.float32)
        if self.frame_pad_len:

            utt_frames = self.pad_right(utt_frames,self.frame_pad_len,num_dims=2,pad_val=0)
        utt_frames = torch.transpose(utt_frames,0,1)
        return utt_frames

    def get_pause(self,iden):
        pause = torch.tensor(self.pause[iden]['pause_aft'],dtype=torch.float32)
        pause = self.pad_right(pause,self.tok_pad_len,num_dims=1,pad_val=0)
        return pause

    def get_dur(self,iden):
        dur = torch.tensor(self.dur[iden],dtype=torch.float32)
        dur = self.pad_right(dur,self.tok_pad_len,num_dims=2,pad_val=0)
        dur = torch.transpose(dur,0,1)
        return dur
    
    def pad_right(self,arr,pad_len,num_dims=2,pad_val=0):
        if arr.shape[0] < pad_len:
            
            dff = pad_len - arr.shape[-1]

            if num_dims==2: # For padding 2d speech data
                #try:
                arr = F.pad(arr, pad=(0, dff, 0, 0), mode='constant',value=pad_val)
                #except:
                #    print(arr.shape)
                #    print(dff)
            elif num_dims==1: # For padding 1d string data
                arr = F.pad(arr, pad=(0, dff), mode='constant',value=pad_val)
        else:
            arr = arr[:pad_len]
        return arr

    def get_toktimes(self,iden):
        """
        return the initial timestamps from the partition
        """
        curr_part = self.partition[iden]
        toktimes = [part[0] for part in curr_part]+[curr_part[-1][-1]]
        toktimes = torch.tensor(toktimes, dtype=torch.float32)
        toktimes = self.pad_right(toktimes,self.tok_pad_len,num_dims=1,pad_val=0)
        return toktimes
    
    def get_labels(self,iden):
        Y = torch.tensor(self.id2lbl[iden],dtype=torch.long)
        
        if self.tok_pad_len:
            Y = self.pad_right(Y, self.tok_pad_len, num_dims=1)
        return Y

    def adjust_vocab_size(self,w2i):
        for wd in w2i:
            if w2i[wd] > self.vocab_size:
                w2i[wd] = w2i['UNK']
        return w2i

    def get_tokens(self,iden):
        toks = torch.tensor(self.id2tokids[iden],dtype=torch.long)
        if self.tok_pad_len:
            toks = self.pad_right(toks,self.tok_pad_len,num_dims=1,pad_val=self.w2i['PAD'])
        return toks

    def load_ids(self):
        with open(self.id_file,'r') as f:
            ids = f.readlines()
        ids = [idnum.strip() for idnum in ids]
        return ids

    def load_lbls(self):
        with open(self.label_file,'r') as f:
            lbls = f.readlines()
        out_lbls = []
        for lbl in lbls:
            line = []
            split_lbl = lbl.strip().split()
            for l in split_lbl:
                line.append(int(l))
            out_lbls.append(line)
        return out_lbls

    def load_text(self):
        with open(self.text_file,'r') as f:
            lines = f.readlines()
        lines = [line.strip().split() for line in lines]
        return lines

    def text2tokid(self):
        tokids = []
        for line in self.text:
            new_line = []
            for wd in line:
                if wd in self.w2i:
                    new_line.append(self.w2i[wd])
                else:
                    new_line.append(self.w2i['UNK'])
            tokids.append(new_line)
        return tokids
    
    def __getitem__(self, index):
        iden = self.ids[index]
        frames = self.get_frames(iden)
        labels = self.get_labels(iden)
        toktimes = self.get_toktimes(iden)
        tok_ints = self.get_tokens(iden)
        pause = self.get_pause(iden)
        dur = self.get_dur(iden)

        return iden, frames,pause,dur, labels, toktimes, tok_ints

