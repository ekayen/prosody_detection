"""
Anonymized
"""
from utils import *
import os
from torch.utils import data
import torch
import numpy as np
import pandas as pd
from random import randint
from sklearn.metrics import precision_recall_fscore_support,f1_score
from torch.nn import functional as F
import time

def evaluate(cfg,
             dataset,
             dataloader_params,
             model,
             device,
             epoch = -1,
             recurrent=True,
             tok_level_pred=False,
             noisy=True,
             text_only=False,
             print_predictions=False,
             vocab_dict=None,
             stopword_list=None,
             stopword_baseline=False,
             #non_default_only=False,
             prf=False,
             maj_baseline=False,
             random_baseline=False,
             prev_best_score=0,
             np_level=False,
             bootstrap_resample=False):

    recurrent = cfg['include_lstm']

    only_eval_np = cfg['only_eval_np'] if 'only_eval_np' in cfg else False
    stopping_score = cfg['stopping_score'] if 'stopping_score' in cfg else 'acc'
        

    predictions = []
    ys = []
    
    model.eval()
    true_pos_pred = 0
    total_pred = 0
    tot_utts = 0
    dataloader = data.DataLoader(dataset, **dataloader_params)
    count_ones = 0 # TODO TMP
    print_lines = []
    with torch.no_grad():
        counter = 0
        for idnum,x,y in dataloader:
            if not text_only: # TODO temporary flag for the text-alone model
                x,text,toktimes = x
                curr_bat_size = x.shape[0]
            else:
                x,toktimes = x
                x = x.transpose(0,1)# TODO temporary fix for text-alone model

                curr_bat_size = x.shape[1]
            tot_utts += curr_bat_size
            x,text,y = x.to(device),text.to(device),y.to(device)
            hidden = model.init_hidden(curr_bat_size)
            #if recurrent:
                #hidden = model.init_hidden(dataloader_params['batch_size'])
            if not text_only: # TODO temporary flag for the text-alone model
                output, _ = model(x, text, toktimes, hidden)
            else:
                output, _ = model(x, hidden)
            #else:
            #    output = model(x, hidden)
            # Stopword baseline here: just labels stopwords/padding 0 and all else 1:
            if stopword_baseline:# or non_default_only:
                #import pdb;pdb.set_trace()
                if not tok_level_pred and cfg['tok_pad_len'] == 3:
                    # For evalling stehwien replication:
                    text = text[:,1]
                    text_list = text.tolist()
                    text_list = [0 if tok in stopword_list else 1 for tok in text_list]
                    content_baseline = torch.tensor(text_list,dtype=torch.int64)
                    content_baseline = torch.unsqueeze(content_baseline,-1)
                else:
                    text_list = text.tolist()
                    text_list = [[0 if tok in stopword_list else 1 for tok in lst] for lst in text_list]
                    content_baseline = torch.tensor(text_list,dtype=torch.int64)
                    
                content_baseline = content_baseline.transpose(0,1)
                content_baseline = content_baseline.view(content_baseline.shape[0],content_baseline.shape[1]).to(device)
                content_baseline = F.one_hot(content_baseline)
                if stopword_baseline:
                    output = content_baseline
            if tok_level_pred:
                seq_len = y.shape[1]

                # num_toks = [np.trim_zeros(np.array(toktimes[i:i + 1]).squeeze(), 'b').shape[0] - 1 for i in
                #            range(toktimes.shape[0])]  # list of len curr_bat_size, each element is len of that utterance
                num_toks = [len(dataset.input_dict['utt2toks'][iden]) for iden in idnum]
             


                #output = output.view(output.shape[0],output.shape[1]).transpose(0, 1)
                output = output.transpose(0, 1)
                output = output.detach().reshape(output.shape[0]*output.shape[1],output.shape[2])
                """
                if non_default_only:
                    content_baseline = content_baseline.view(content_baseline.shape[0],
                                                             content_baseline.shape[1]).transpose(0, 1).flatten()
                """
                
                #output = output.detach().flatten()

                y = y.flatten()

                #import pdb;pdb.set_trace()

                tmp_out = []
                tmp_lbl = []
                tmp_cont_baseline = []
                tmp_np_mask = []
                for i in range(curr_bat_size):
                    start = i * seq_len
                    end = i * seq_len + num_toks[i] 
                    if only_eval_np: # only count tokens that are part of NPs toward score
                        tmp_out.append(output[start:end])
                        curr_lbl = y[start:end]
                        tmp_lbl.append(curr_lbl)
                        mask = torch.tensor([0 if j=='O' else 1 for j in dataset.input_dict['utt2bio'][idnum[i]]])
                        tmp_np_mask.append(mask)
                    else:
                        curr_lbl = y[start:end]
                        try:
                            #assert curr_lbl.shape[0] == len(dataset.input_dict['utt2nps'][idnum[i]])
                            #assert curr_lbl.shape[0] == len(dataset.input_dict['utt2old'][idnum[i]])
                            assert curr_lbl.shape[0] == len(dataset.input_dict['utt2toks'][idnum[i]])
                        except AssertionError:
                            print('mis')

                        tmp_out.append(output[start:end])
                        tmp_lbl.append(curr_lbl)
                    """
                    if non_default_only:
                        tmp_cont_baseline.append(content_baseline[(i * seq_len):(i * seq_len) + num_toks[i]])
                    """
                out = torch.cat(tmp_out)
                lbl = torch.cat(tmp_lbl)
                if only_eval_np:
                    long_out = out # In order to print out preds, we need to keep the full predictions
                    long_lbl = lbl
                    np_mask = torch.cat(tmp_np_mask)
                    out = out[np_mask>0]
                    lbl = lbl[np_mask>0]

                output = out
                y = lbl

            else:
                # TODO 2024 the problem here is that my output is shaped (3, 64, 2) and it should be (64, 2).
                # Looks like I'm predicting for each token, and that's not right.
                #print(f'output shape in eval: {output.shape}')
                #import pdb;pdb.set_trace()
                output = output.detach().view(output.shape[-2],output.shape[-1])

            prediction = output.argmax(-1)
            if maj_baseline:
                #prediction = torch.zeros(prediction.shape,dtype=torch.long).to(device)
                prediction = torch.ones(prediction.shape,dtype=torch.long).to(device)
            elif random_baseline:
                prediction = torch.randint(0,cfg['num_classes'],prediction.shape).to(device)
                #a = torch.empty(prediction.shape).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
                #prediction = torch.bernoulli(a).to(device)

            elif np_level: # treat NPs as single units when scoring

                print('np_level',np_level)

                np_bio = []
                for i in range(curr_bat_size):
                    np_bio.extend(dataset.input_dict['utt2nps'][idnum[i]])
                    try:
                        assert len(dataset.input_dict['utt2nps'][idnum[i]])==len(dataset.input_dict['utt2old'][idnum[i]])
                    except AssertionError:
                        print(idnum[i])
                        import pdb;pdb.set_trace()
                    try:
                        assert len(np_bio)==y.shape[0]
                    except AssertionError:
                        print('len mismatch')
                        import pdb;pdb.set_trace()

                shortened_prediction = []
                shortened_y = []
                tmp_np = []
                try:
                    assert len(np_bio) == prediction.shape[0]
                except:
                    import pdb;pdb.set_trace()
                for j,tok in enumerate(np_bio):
                    if tok=='O':
                        if tmp_np:
                            # Collapse down the NP that you have accumulated the idxs for in tmp_np
                            # This can be done just by taking the max of those
                            shortened_prediction.append(torch.max(prediction[tmp_np[0]:tmp_np[-1]+1]))
                            shortened_y.append(torch.max(y[tmp_np[0]:tmp_np[-1]+1]))

                            tmp_np = []
                        shortened_prediction.append(prediction[j])
                        shortened_y.append(y[j])
                    elif tok=='B':
                        if tmp_np:
                            shortened_prediction.append(torch.max(prediction[tmp_np[0]:tmp_np[-1]+1]))
                            shortened_y.append(torch.max(y[tmp_np[0]:tmp_np[-1]+1]))
                            tmp_np = []
                        tmp_np.append(j)
                    elif tok=='I':
                        tmp_np.append(j)
                prediction = torch.tensor(shortened_prediction)
                y = torch.tensor(shortened_y)

            predictions.append(prediction)
            ys.append(y)
            if only_eval_np:
                long_prediction = long_out.argmax(-1) # Also generate full preds, not just the part we eval
            #print(prediction.sum())

            if print_predictions:

                sents = [np.trim_zeros(np.array(text[i:i+1].flatten().cpu()),'b').tolist() for i in range(curr_bat_size)]
                lens = [len(sent) for sent in sents]
                if only_eval_np:
                    prediction_list = long_prediction.tolist()
                else:
                    prediction_list = prediction.tolist()
                    
                lbl_list = y.tolist()
                preds = []
                lbls = []
                for i,sent in enumerate(sents):
                    curr_len = len(sent)
                    preds.append([str(j) for j in prediction_list[0:curr_len]])
                    lbls.append([str(j) for j in lbl_list[0:curr_len]])
                    prediction_list = prediction_list[curr_len:]
                    lbl_list = lbl_list[curr_len:]

                print_lines.extend([(sents[i],lbls[i],preds[i],idnum[i]) for i in range(len(sents))])

                
            count_ones += prediction.sum().item()

            if tok_level_pred:
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
    if bootstrap_resample:
        boot_accs = []

        from random import choices
        num_utt = len(predictions)
        for i in range(1000):
            curr_preds = []
            curr_lbls = []
            idxes = random.choices(list(range(num_utt)),k=num_utt)
            for idx in idxes:
                curr_preds.append(predictions[idx].cpu())
                curr_lbls.extend(ys[idx])
            curr_pred = np.concatenate(curr_preds)
            curr_lbl = np.array([int(l) for l in curr_lbls])
            total_pred = curr_pred.shape[0]
            true_pos_pred = (curr_pred==curr_lbl).sum().item()
            boot_acc = (true_pos_pred/total_pred)
            boot_accs.append(boot_acc)
        boot_accs = np.array(boot_accs)
        print(f'bootstrap_mean: {np.mean(boot_accs)}')
        print(f'bootstrap_std: {np.std(boot_accs)}')

            
    """
    if non_default_only:
        precision,recall,fscore,support = precision_recall_fscore_support(prediction.cpu(),y.cpu())
        #print()
        #print(f'precision: {precision}')
        #print(f'recall: {recall}')
        #print(f'support: {support}')
        return acc, total_pred, true_pos_pred, tot_utts, precision[0], precision[1], recall[0], recall[1]
    """
    if prf:
        all_preds = torch.cat(predictions)
        all_y = torch.cat(ys)
        all_preds = all_preds.cpu()
        all_y = all_y.cpu()
        precision,recall,fscore,support = precision_recall_fscore_support(all_preds,all_y)
        if noisy:
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F-score: {fscore}')
        #print(f'Micro f1:', f1_score(all_y,all_preds,average='micro'))
        #print(f'Macro f1:', f1_score(all_y,all_preds,average='macro'))
    
        if print_predictions:
            model_name = gen_model_name(cfg,cfg['datasplit'])
            if (stopping_score == 'acc' and acc > prev_best_score) or \
               (stopping_score == 'f' and max(fscore) > prev_best_score):
                with open(os.path.join(cfg['results_path'],f'{model_name}.pred.{epoch}'),'w') as f:
                    f.write(f'ID\ttext\tlabels\tpredicted_labels\n')
                    for line in print_lines:
                        sent = [vocab_dict['i2w'][i] for i in line[0]]
                        f.write(f'{line[-1]}\t{" ".join(sent)}\t{" ".join(line[1])}\t{" ".join(line[2])}')
                        f.write('\n')
                     
        return acc, total_pred, true_pos_pred, tot_utts, precision, recall, fscore
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
        for idnum, x, y in dataloader:
            idnum = idnum[0]
            utt_len = key_to_len[idnum]
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
    for idnum,x,y in dataloader:
        idnum = idnum[0]
        if key_to_len[idnum] > 1:
            pred = randint(0,1)
        else:
            pred = 0
        total_pred += 1
        if pred == y.item():
            true_pos_pred += 1
        acc = true_pos_pred/total_pred
    print('Baseline performance with length considered:',acc)

