import pandas as pd
import matplotlib.pyplot as plt

span_pred = False
incl_true = True

#predfile = '../../../../data/burnc/text2labels_breath_tok'
#predfile = 'train_w_true_labels.tsv'
predfile = 'lstm_predictions_w_true.tsv'

df = pd.read_csv(predfile,sep='\t',header=None)

words = df[0].tolist()
labels = df[1].tolist()
if incl_true: true_labels = df[2]

if span_pred:
    words = [[w2 for w1,w2,w3 in line] for line in words]
    labels = [[lbl] for lbl in labels]
    if incl_true: true_labels = [[lbl] for lbl in labels]

tok2count = {}
tok2numones = {}
tok2trueones = {}
if not incl_true:
    true_labels = [0 for i in words]

for toks,lbls,true_lbls in zip(words,labels,true_labels):
    toks = toks.split()
    lbls = lbls.split()
    if incl_true:
        true_lbls = true_lbls.split()
    else:
        true_lbls = [0 for i in toks]

    for tok,lbl,true_lbl in (zip(toks,lbls,true_lbls)):
        if tok in tok2count:
            tok2count[tok] += 1
        else:
            tok2count[tok] = 1
        if lbl=='1':
            if tok in tok2numones:
                tok2numones[tok] += 1
            else:
                tok2numones[tok] = 1

        if incl_true and true_lbl=='1':
            if tok in tok2trueones:
                tok2trueones[tok] += 1
            else:
                tok2trueones[tok] = 1

all_toks = set(tok2count.keys())
for tok in all_toks:
    if not tok in tok2numones:
        tok2numones[tok] = 0
    if incl_true:
        if not tok in tok2trueones:
            tok2trueones[tok] = 0

tok2percent = {}
for tok in tok2count:
    tok2percent[tok] = tok2numones[tok]/tok2count[tok]

if incl_true:
    tok2truepercent = {}
    for tok in tok2count:
        tok2truepercent[tok] = tok2trueones[tok]/tok2count[tok]


# make a hist of the percents
tokpercent = list(zip(list(tok2percent.keys()),list(tok2percent.values())))
tokpercent.sort(key=lambda x: x[1])
percents = [i for tok,i in tokpercent]

plt.hist(percents,color='blue')



plt.xlabel('Percentage of predictions that are 1')
plt.ylabel('Number of types')
plt.show()

# Check which tokens get no 1 guesses

all_ones = [tok for tok,i in tokpercent if i==1]
all_zeros =[tok for tok,i in tokpercent if i==0]
others = [tok for tok in tok2count if not tok in all_ones]

print('num types',len(all_toks))
print('num all ones:',len(all_ones)/len(all_toks))
print('num all zeros:',len(all_zeros)/len(all_toks))
freq_of_all_ones = [tok2count[tok] for tok in all_ones]
freq_of_others = [tok2count[tok] for tok in others]

# make a scatterplot of freq vs percent
plt.xlabel('Frequency of type')
plt.ylabel('Percentage of predictions that are 1')

freqs = [tok2count[tok] for tok,per in tokpercent]
plt.scatter(freqs,percents)
plt.show()


if incl_true:
    # make a hist of the percents
    toktruepercent = list(zip(list(tok2truepercent.keys()), list(tok2truepercent.values())))
    toktruepercent.sort(key=lambda x: x[1])
    truepercents = [i for tok, i in toktruepercent]

    plt.hist(truepercents, color='blue')

    plt.xlabel('Percentage of labels that are 1')
    plt.ylabel('Number of types')
    plt.show()

    # Check which tokens get no 1 guesses
    true_all_ones = [tok for tok, i in toktruepercent if i == 1]
    true_all_zeros = [tok for tok, i in toktruepercent if i == 0]
    print('num true all ones:', len(true_all_ones)/len(all_toks))
    print('num true all zeros:', len(true_all_zeros)/len(all_toks))
    true_freq_of_all_ones = [tok2count[tok] for tok in true_all_ones]
    true_others = [tok for tok in tok2count if not tok in true_all_ones]
    true_freq_of_others = [tok2count[tok] for tok in true_others]

    # make a scatterplot of freq vs percent
    plt.xlabel('Frequency of type')
    plt.ylabel('Percentage of labels that are 1')

    freqs = [tok2count[tok] for tok, per in toktruepercent]
    plt.scatter(freqs, truepercents)
    plt.show()

import pdb;pdb.set_trace()