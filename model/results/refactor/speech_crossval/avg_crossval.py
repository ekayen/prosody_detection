import os
import sys
import pandas as pd
import argparse
import glob


metric = 'train_losses'


target_dir = sys.argv[1]
if len(sys.argv) > 2:
    filter = sys.argv[2]
else:
    filter = '.tsv'



print(target_dir)
max_dev_accs = []
max_train_accs = []
min_train_losses = []
#for file in glob.glob(target_dir):
for file in os.listdir(target_dir):
    #import pdb;pdb.set_trace()
    if file.endswith(".tsv"):

        if filter in file:

            print(file)
            df = pd.read_csv(file, sep='\t')
            dev_accs = df['dev_accs'].tolist()
            train_accs = df['train_accs'].tolist()
            train_losses = df['train_losses'].tolist()

            max_dev_accs.append(max(dev_accs))
            max_train_accs.append(max(train_accs))
            min_train_losses.append(min(train_losses))


print(f'Average of best:\t train_loss: {sum(min_train_losses)/len(min_train_losses)}\t train_acc: {sum(max_train_accs)/len(max_train_accs)}\t dev_acc:{sum(max_dev_accs)/len(max_dev_accs)}\t')
