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
base_call = f"python3 train.py -c conf/cnn_lstm_pros.yaml"

learning_rates = [1e-4, 1e-3, 1e-2]
weight_decays = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
lstm_layers = [2,3]
dropout = [0,0.2,0.5]
hidden_size = [128,256,512]
bottleneck_feats = [10,70,100,700,1000,1500,2000,2500,3000]


num_experiments = 50


output_base = 'both_exp'

random.seed(29)

per_file = 6

calls = []

for exp in range(num_experiments):
    lr_idx = random.randrange(len(learning_rates))
    wd_idx = random.randrange(len(weight_decays))
    lstm_idx = random.randrange(len(lstm_layers))
    dr_idx = random.randrange(len(dropout))
    hid_idx = random.randrange(len(hidden_size))
    btl_idx = random.randrange(len(bottleneck_feats))


    expt_call = (
        f"{base_call} "
        f"-lr {learning_rates[lr_idx]} "
        f"-wd {weight_decays[wd_idx]} "
        f"-l {lstm_layers[lstm_idx]} "
        f"-dr {dropout[dr_idx]} "
        f"-hid {hidden_size[hid_idx]} "
        f"-b {bottleneck_feats[btl_idx]} "
    )
    calls.append(expt_call)
#    print(expt_call, file=output_file)


calls = [calls[i:i + per_file] for i in range(0, len(calls), per_file)]

for i,call in enumerate(calls):
    filename = f'{output_base}{i}.sh'
    print_call = ' &\n'.join(call)
    with open(filename,'w') as f:
        f.write(print_call)
