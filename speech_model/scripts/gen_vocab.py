#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
import random

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'  
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/stars/data'
base_call = f"python3 train.py -c conf/cnn_lstm_best.yaml"

vocab_sizes = [3000,2500,2000,1500,1000,750,500,250,100,50,25,10,5]

output_base = 'both_vocab'

random.seed(29)

per_file = 6

calls = []

for size in vocab_sizes:

    expt_call = (
        f"{base_call} "
        f"-v {size} "
    )
    calls.append(expt_call)
#    print(expt_call, file=output_file)


calls = [calls[i:i + per_file] for i in range(0, len(calls), per_file)]

for i,call in enumerate(calls):
    filename = f'{output_base}{i}.sh'
    print_call = ' &\n'.join(call)
    with open(filename,'w') as f:
        f.write(print_call)
