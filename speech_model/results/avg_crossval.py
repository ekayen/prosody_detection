import os
import sys
import pandas as pd
import argparse
import glob

#metric = 'dev_accs'
#metric = 'train_accs'
metric = 'train_losses'

target_dir = sys.argv[1]
print(target_dir)
max_accs = []
for file in glob.glob(target_dir):
    if file.endswith(".tsv"):
        print(file)
        df = pd.read_csv(file, sep='\t')
        dev_accs = df[metric].tolist()
        if metric=='train_losses':
            max_accs.append(min(dev_accs))
        else:
            max_accs.append(max(dev_accs))


print('Average of best performances:')
print(sum(max_accs)/len(max_accs))

