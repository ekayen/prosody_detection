import numpy as np
import pickle
import torch
from seqeval.metrics import accuracy_score, classification_report,f1_score


def evaluate(X, Y,mdl,device,to_file=False):
    print("EVAL=================================================================================================")
    y_pred = []
    with torch.no_grad():
        for i in range(len(X)):
            input = X[i].to(device)
            eval_batch_size = 1

            if not (list(input.shape)[0] == 0):
                hidden = mdl.init_hidden(eval_batch_size)
                tag_scores, _ = mdl(input.view(len(input),eval_batch_size), hidden)


                pred = tag_scores.cpu()
                pred = np.where(pred>0.5,1,0)
                pred = np.squeeze(pred)
                pred = pred.tolist()
                if type(pred) is int:
                    pred = [pred]
                pred = [str(j) for j in pred]

                y_pred.append(pred)
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