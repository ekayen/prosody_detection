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
#for file in glob.glob(target_dir):
for file in os.listdir(target_dir):
    if file.endswith(".tsv") and subset in file:
        print(file)
        df = pd.read_csv(file, sep='\t')
        dev_accs = df['dev_accs'].tolist()
        train_accs = df['train_accs'].tolist()
        train_losses = df['train_losses'].tolist()
        
        max_dev_accs.append(max(dev_accs))
        max_train_accs.append(max(train_accs))
        min_train_losses.append(min(train_losses))


print(f'Average of best:\t train_loss: {sum(min_train_losses)/len(min_train_losses)}\t train_acc: {sum(max_train_accs)/len(max_train_accs)}\t dev_acc:{sum(max_dev_accs)/len(max_dev_accs)}\t')

from scipy.stats import sem, t
from scipy import mean

confidence = 0.95

data = max_dev_accs

n = len(data)
m = mean(data)
std_err = sem(data)
h = std_err * t.ppf((1 + confidence) / 2, n - 1)

start = m - h
end = m + h
print(f'95 % confidence interval: {start}, {end}')
print(f'standard error: {std_err}')
