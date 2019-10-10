import pickle
import os
import numpy as np

def print_vocab(prefix):
    with open(os.path.join(prefix,'idx.pkl'),'rb') as f:
        wd_to_i,i_to_wd = pickle.load(f)
    print("Vocab left: ",' | '.join(list(wd_to_i.keys())))

def do_ana(prefix):
    with open(os.path.join(prefix, 'X.pkl'), 'rb') as f:
        X = pickle.load(f)
    X = [x.tolist() for x in X]

    flat_X = [item for sublist in X for item in sublist]

    percent_unk = flat_X.count(0)/len(flat_X)
    print('Percent of types replaced by UNK:',percent_unk)

    with open(os.path.join(prefix,'Y_true.pkl'),'rb') as f:
        Y_true = pickle.load(f)

    Y_true = [[int(y) for y in row] for row in Y_true]

    with open(os.path.join(prefix,'Y_pred.pkl'),'rb') as f:
        Y_pred = pickle.load(f)

    Y_pred = [[int(y) for y in row] for row in Y_pred]

    Y_unk_pred = []
    num0 = 0
    num1 = 0
    for x,y in zip(X,Y_pred):
        x = np.array(x)
        y = np.array(y)
        y = y[x==0]
        num1 += np.sum(y)
        num0 += (y.shape[0] - np.sum(y))

    print('Percent of UNKs labeled 0: ', (num0 / (num1 + num0)))
    print('Percent of UNKs labeled 1: ', (num1 / (num1 + num0)))


print("Vocab size: ",4000)
do_ana('4000')
print()
print("Vocab size: ",100)
do_ana('100')
print_vocab('100')
print()
print("Vocab size: ",50)
do_ana('50')
print_vocab('50')
print()
print("Vocab size: ",10)
do_ana('10')
print_vocab('10')




