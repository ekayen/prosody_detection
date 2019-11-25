from utils import UttDataset
from torch.utils import data
import torch
import numpy as np
import pandas as pd
from random import randint

def evaluate(dataset,dataloader_params,model,device,recurrent=True,tok_level_pred=False):
    model.eval()
    true_pos_pred = 0
    total_pred = 0
    dataloader = data.DataLoader(dataset, **dataloader_params)
    with torch.no_grad():
        counter = 0
        for id,x,y in dataloader:
            x,toktimes = x
            if x.shape[0]==dataloader_params['batch_size']:
                x,y = x.to(device),y.to(device)
                if recurrent:
                    hidden = model.init_hidden(dataloader_params['batch_size'])
                    output,_ = model(x,toktimes,hidden)
                else:
                    output = model(x)
                #print('output shape:',output.shape)
                if tok_level_pred:
                    output = output.detach().view(dataloader_params['batch_size'],output.shape[0])
                else:
                    output = output.detach().view(output.shape[-2])

                threshold = 0
                prediction = (output > threshold).type(torch.int64) * 1
                #prediction = torch.tensor(prediction,dtype=torch.int64)
                if tok_level_pred:
                    total_pred += prediction.shape[1]
                    true_pos_pred += (prediction == y).int().sum().item()
                else:
                    total_pred += prediction.shape[0]
                    true_pos_pred += (prediction == y).int().sum().item()
            else:
                print('rejected last small batch of size',x.shape)
    acc = true_pos_pred/total_pred
    print('Accuracy: ',acc)
    return acc

def evaluate_lengths(dataset,dataloader_params,model,device,recurrent=True,utterance_file='../data/utterances.txt'):
    model.eval()
    true_pos_pred = 0
    total_pred = 0
    dataloader = data.DataLoader(dataset, **dataloader_params)
    # To evaluate if it's using length as a signal, load a dict of the
    # keys to the lengths of the utterances (in num of tokens)
    utterance_df = pd.read_csv(utterance_file,sep='\t',header=None)
    keys = utterance_df[1].tolist()
    tokens = utterance_df[2].tolist()
    tok_len = [len(line.split()) for line in tokens]
    key_to_len = dict(zip(keys,tok_len))
    # Two dictionaries for calculating distribution of predictions
    # by length: one with just the zero preds, one with all preds. Key is length.
    zero_pred_per_len = {}
    pred_per_len = {}

    with torch.no_grad():
        for id, x, y in dataloader:
            id = id[0]
            utt_len = key_to_len[id]
            x,y = x.to(device),y.to(device)
            if recurrent:
                hidden = model.init_hidden(dataloader_params['batch_size'])
                output,_ = model(x,hidden)
            else:
                output = model(x)
            prediction = 1 if output.item() > 0 else 0
            total_pred += 1
            if prediction == y.item():
                true_pos_pred += 1
            if prediction == 0:
                if utt_len in zero_pred_per_len:
                    zero_pred_per_len[utt_len] += 1
                else:
                    zero_pred_per_len[utt_len] = 1
            if utt_len in pred_per_len:
                pred_per_len[utt_len] += 1
            else:
                pred_per_len[utt_len] = 1


    percent_pred_per_len = {}
    for l in pred_per_len:
        if l in zero_pred_per_len:
            percent_pred_per_len[l] = zero_pred_per_len[l]/pred_per_len[l]
        else:
            percent_pred_per_len[l] = 0
    print("Percent of single-token utterances labeled zero: ",percent_pred_per_len[1])
    acc = true_pos_pred/total_pred
    print('Accuracy: ',acc)
    return acc



def baseline_with_len(dataset,dataloader_params,utterance_file='../data/utterances.txt'):
    # To evaluate if it's using length as a signal, load a dict of the
    # keys to the lengths of the utterances (in num of tokens)

    utterance_df = pd.read_csv(utterance_file,sep='\t',header=None)
    keys = utterance_df[1].tolist()
    tokens = utterance_df[2].tolist()
    tok_len = [len(line.split()) for line in tokens]
    key_to_len = dict(zip(keys,tok_len))

    true_pos_pred = 0
    total_pred = 0

    dataloader = data.DataLoader(dataset, **dataloader_params)
    for id,x,y in dataloader:
        id = id[0]
        if key_to_len[id] > 1:
            pred = randint(0,1)
        else:
            pred = 0
        total_pred += 1
        if pred == y.item():
            true_pos_pred += 1
        acc = true_pos_pred/total_pred
    print('Baseline performance with length considered:',acc)

