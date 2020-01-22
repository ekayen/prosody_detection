from utils import *
import os
from torch.utils import data
import torch
import numpy as np
import pandas as pd
from random import randint

def evaluate(cfg,dataset,dataloader_params,model,device,recurrent=True,tok_level_pred=False,noisy=True,text_only=False,
             print_predictions=False,vocab_dict=None):
    model.eval()
    true_pos_pred = 0
    total_pred = 0
    tot_utts = 0
    dataloader = data.DataLoader(dataset, **dataloader_params)
    count_ones = 0 # TODO TMP
    print_lines = []
    with torch.no_grad():
        counter = 0
        for id,x,y in dataloader:
            if not text_only: # TODO temporary flag for the text-alone model
                x,text,toktimes = x
                curr_bat_size = x.shape[0]
            else:
                x,toktimes = x
                x = x.transpose(0,1)# TODO temporary fix for text-alone model

                curr_bat_size = x.shape[1]
            tot_utts += curr_bat_size
            x,text,y = x.to(device),text.to(device),y.to(device)
            if recurrent:
                #hidden = model.init_hidden(dataloader_params['batch_size'])
                hidden = model.init_hidden(curr_bat_size)
                if not text_only: # TODO temporary flag for the text-alone model
                    output, _ = model(x, text, toktimes, hidden)
                else:
                    output, _ = model(x, hidden)

            else:
                output = model(x)
            #print('output shape:',output.shape)
            if tok_level_pred:
                seq_len = y.shape[1]

                num_toks = [np.trim_zeros(np.array(toktimes[i:i + 1]).squeeze(), 'b').shape[0] - 1 for i in
                            range(toktimes.shape[0])]  # list of len curr_bat_size, each element is len of that utterance

                #import pdb;pdb.set_trace()
                output = output.view(output.shape[0],output.shape[1]).transpose(0, 1)

                output = output.detach().flatten()
                y = y.flatten()

                tmp_out = []
                tmp_lbl = []
                for i in range(curr_bat_size):
                    tmp_out.append(output[(i * seq_len):(i * seq_len) + num_toks[i]])
                    tmp_lbl.append(y[(i * seq_len):(i * seq_len) + num_toks[i]])
                out = torch.cat(tmp_out)
                lbl = torch.cat(tmp_lbl)

                assert lbl.sum().item() == y.sum().item()

                output = out
                y = lbl

            else:
                output = output.detach().view(output.shape[-2])

            threshold = 0
            prediction = (output > threshold).type(torch.int64) * 1

            if print_predictions:

                sents = [np.trim_zeros(np.array(text[i:i+1].flatten().cpu()),'b').tolist() for i in range(curr_bat_size)]
                lens = [len(sent) for sent in sents]

                prediction_list = prediction.tolist()
                lbl_list = y.tolist()
                preds = []
                lbls = []
                for i,sent in enumerate(sents):
                    preds.append([str(j) for j in prediction_list[0:lens[i]]])
                    lbls.append([str(j) for j in lbl_list[0:lens[i]]])
                    prediction_list = prediction_list[lens[i]:]
                    lbl_list = lbl_list[lens[i]:]

                print_lines.extend([(sents[i],lbls[i],preds[i]) for i in range(len(sents))])

            count_ones += prediction.sum().item()

            #prediction = torch.tensor(prediction,dtype=torch.int64)
            if tok_level_pred:
                #total_pred += prediction.shape[1]
                total_pred += prediction.shape[0]
                true_pos_pred += (prediction == y).int().sum().item()

            else:
                total_pred += prediction.shape[0]
                true_pos_pred += (prediction == y).int().sum().item()


    acc = true_pos_pred/total_pred
    if noisy:
        print(f'Accuracy: {round(acc,5)}')
        print(f'Total_pred: {total_pred}')
        print(f'Total correct pred: {true_pos_pred}')
        print(f'Percent of predicted "1" labels: {count_ones/total_pred}')
    if print_predictions:
        model_name = gen_model_name(cfg,cfg['datasplit'])
        with open(os.path.join(cfg['results_path'],f'{model_name}.pred'),'w') as f:
            f.write(f'text\tlabels\tpredicted_labels\n')
            for line in print_lines:
                sent = [vocab_dict['i2w'][i] for i in line[0]]
                f.write(f'{" ".join(sent)}\t{" ".join(line[1])}\t{" ".join(line[2])}')
                f.write('\n')
    return acc, total_pred, true_pos_pred, tot_utts

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

