import os
import sys
import pandas as pd
import argparse
import glob

metric = 'train_losses'

target_dir = sys.argv[1]
subset = sys.argv[2]

print(target_dir)
max_dev_accs = []
max_train_accs = []
min_train_losses = []
non_default_accs = []
non_default_precision_0 = []
non_default_precision_1 = []
non_default_recall_0 = []
non_default_recall_1 = []
#for file in glob.glob(target_dir):
for file in os.listdir(target_dir):
    if file.endswith(".tsv") and subset in file:
        print(file)
        df = pd.read_csv(file, sep='\t')
        dev_accs = df['dev_accs'].tolist()
        train_accs = df['train_accs'].tolist()
        train_losses = df['train_losses'].tolist()
        
        max_dev_accs.append(max(dev_accs))

        if 'non_default_accs' in df.columns:

            row_idx = df.index[df['dev_accs'] == max_dev_accs[-1]]
            #import pdb;pdb.set_trace()

            nd_acc = next(iter(df.iloc[row_idx]['non_default_accs']))
            non_default_accs.append(nd_acc)

            nd_prec_0 = next(iter(df.iloc[row_idx]['non_default_precision_0']))
            non_default_precision_0.append(nd_prec_0)

            nd_prec_1 = next(iter(df.iloc[row_idx]['non_default_precision_1']))
            non_default_precision_1.append(nd_prec_1)

            nd_rec_0 = next(iter(df.iloc[row_idx]['non_default_recall_0']))
            non_default_recall_0.append(nd_rec_0)

            nd_rec_1 = next(iter(df.iloc[row_idx]['non_default_recall_1']))
            non_default_recall_1.append(nd_rec_1)



        max_train_accs.append(max(train_accs))
        min_train_losses.append(min(train_losses))

#import pdb;pdb.set_trace()

print(sorted(max_train_accs))

print(f'Average of best:\t train_loss: {sum(min_train_losses)/len(min_train_losses)}\t train_acc: {sum(max_train_accs)/len(max_train_accs)}\t dev_acc:{sum(max_dev_accs)/len(max_dev_accs)}\t')
if 'non_default_accs' in df.columns:
    print(f'Average of non-default accuracy: {sum(non_default_accs) / len(non_default_accs)}')
    print(f'Average of non-default precision on class 0: {sum(non_default_precision_0) / len(non_default_precision_0)}')
    print(f'Average of non-default precision on class 1: {sum(non_default_precision_1) / len(non_default_precision_1)}')
    print(f'Average of non-default recall on class 0: {sum(non_default_recall_0) / len(non_default_recall_0)}')
    print(f'Average of non-default recall on class 1: {sum(non_default_recall_1) / len(non_default_recall_1)}')

from scipy.stats import sem, t,tvar,tstd,tmean,tsem
from scipy import mean

confidence = 0.95

data = max_dev_accs

n = len(data)
m = mean(data)
std_err = sem(data)
h = std_err * t.ppf((1 + confidence) / 2, n - 1)
var = tvar(data)
stddev = tstd(data)

start = m - h
end = m + h
print(f'95 % confidence interval: {start}, {end}')
print(f'standard error: {std_err}')
print(f'variance: {var}')
print(f'standard dev: {stddev}')


print(f'trimmed mean: {tmean(data,(0.6,1))}')
print(f'trimmed variance: {tvar(data,(0.6,1))}')
print(f'trimmed stderr: {tsem(data,(0.6,1))}')
