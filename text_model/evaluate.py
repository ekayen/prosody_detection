import numpy as np
import pickle
import torch
from seqeval.metrics import accuracy_score, classification_report,f1_score
import sys
import yaml
import os

if len(sys.argv) < 2:
    config = 'conf/burnc.yaml'
else:
    config = sys.argv[1]

with open(config,'r') as f:
    cfg = yaml.load(f,yaml.FullLoader)

results_path = cfg['results_path']

def write_preds(output_preds,timestep,i_to_wd):
    filename = 'predictions'
    if timestep:
        filename += str(timestep)
    filename += '.tsv'
    filepath = os.path.join(results_path,filename)
    with open(filepath,'w') as f:
        f.write('input' + '\t' + 'prediction' + '\t' + 'true' + '\n')
        for x,y_pred,y in output_preds:
            x = [i_to_wd[i] for i in x.tolist()]
            if type(y_pred)==int:
                y_pred = [y_pred]
            y_pred = [str(i) for i in y_pred]
            f.write(' '.join(x)+'\t'+' '.join(y_pred)+'\t'+' '.join(y)+'\n')

def evaluate(X, Y,mdl,device,i_to_wd,model_type='lstm',to_file=False,print_preds=False,timestep=None):
    print("EVAL=================================================================================================")
    y_pred = []
    output_preds = []
    with torch.no_grad():
        for i in range(len(X)):
            input = X[i].to(device)
            eval_batch_size = 1

            if not (list(input.shape)[0] == 0):
                if model_type=='lstm':
                    hidden = mdl.init_hidden(eval_batch_size)
                    tag_scores, _ = mdl(input.view(len(input),eval_batch_size), hidden)
                else:
                    tag_scores = mdl(input.view(len(input),eval_batch_size))

                pred = tag_scores.cpu()
                pred = np.where(pred>0.5,1,0)
                pred = np.squeeze(pred)
                pred = pred.tolist()
                if print_preds:
                    output_preds.append((input,pred,Y[i]))
                if type(pred) is int:
                    pred = [pred]
                pred = [str(j) for j in pred]

                y_pred.append(pred)
    write_preds(output_preds,timestep,i_to_wd)
    print('Evaluation:')

    if to_file:
        with open('Y_true.pkl','wb') as f:
            pickle.dump(Y,f)
        with open('Y_pred.pkl','wb') as f:
            pickle.dump(y_pred,f)
        with open('X.pkl','wb') as f:
            pickle.dump(X,f)

    f1 = f1_score(Y, y_pred)
    acc = accuracy_score(Y, y_pred)
    clss = classification_report(Y, y_pred)
    print('F1:',f1)
    print('Acc',acc)
    print(clss)
    return(f1,acc,clss)

def nonseq_evaluate(X,Y,mdl,device):
    true_pos_pred =0
    total_pred = 0
    y_pred = []
    with torch.no_grad():
        for i in range(len(X)):
            input = X[i].to(device)
            true_label = Y[i]
            eval_batch_size = 1
            if not (list(input.shape)[0] == 0):
                hidden = mdl.init_hidden(eval_batch_size)
                tag_scores, _ = mdl(input.view(len(input),eval_batch_size), hidden)

                pred = tag_scores.detach().cpu().squeeze()
                pred = 1 if pred.item() > 0.5 else 0
                if pred == true_label:
                    true_pos_pred += 1
                total_pred += 1

    acc = true_pos_pred/total_pred
    print('Acc:',acc)
    return acc

def last_only_evaluate(X,Y,mdl,device):
    true_pos_pred = 0
    total_pred = 0
    with torch.no_grad():
        for i in range(len(X)):
            input = X[i].to(device)
            true_label = int(Y[i][-1])
            eval_batch_size = 1
            if not (list(input.shape)[0] == 0):
                hidden = mdl.init_hidden(eval_batch_size)
                tag_scores, _ = mdl(input.view(len(input), eval_batch_size), hidden)

                pred = tag_scores.detach().cpu().squeeze().tolist()
                if type(pred) is list:
                    pred = pred[-1]
                pred = 1 if pred > 0.5 else 0
                if pred == true_label:
                    true_pos_pred += 1
                total_pred += 1

    acc = true_pos_pred / total_pred
    print('Acc:', acc)
    return acc
