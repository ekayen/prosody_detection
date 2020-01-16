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
base_call = f"python3 train.py -c conf/cnn_lstm_cluster.yaml"

learning_rates = [1e-4, 1e-3, 1e-2]
weight_decays = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
lstm_layers = [2,3]
dropout = [0,0.2,0.5]
bottleneck_feats = [10,70,100,700,1000,1500,2000]
embedding_dim = [100,300]

num_experiments = 50


output_file = open("text_exp.sh", "w")

random.seed(40)

for exp in range(num_experiments):
    lr_idx = random.randrange(len(learning_rates))
    wd_idx = random.randrange(len(weight_decays))
    lstm_idx = random.randrange(len(lstm_layers))
    dr_idx = random.randrange(len(dropout))
    btl_idx = random.randrange(len(bottleneck_feats))
    
    expt_call = (
        f"{base_call} "
        f"-lr {learning_rates[lr_idx]} "
        f"-wd {weight_decays[wd_idx]} "
        f"-l {lstm_layers[lstm_idx]} "
        f"-dr {dropout[dr_idx]} "
        f"-b {bottleneck_feats[btl_idx]}"
    )
    print(expt_call, file=output_file)

output_file.close()
