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
flatten_methods = ['sum','max']
lstm_layers = [2,3]
cnn_layers = [2,3,4]
dropout = [0,0.2,0.5]
frame_filter_size = [9,11,13,15,17,19,21,23]
frame_pad_size = [4,5,6,7,8,9,10,11]
hidden_size = [128,256,512]
bottleneck_feats = [10,70,100,700,1000,1500,2000]

num_experiments = 12


output_file = open("experiment13.txt", "w")

random.seed(700)

for exp in range(num_experiments):
    lr_idx = random.randrange(len(learning_rates))
    wd_idx = random.randrange(len(weight_decays))
    fl_idx = random.randrange(len(flatten_methods))
    lstm_idx = random.randrange(len(lstm_layers))
    cnn_idx = random.randrange(len(cnn_layers))
    dr_idx = random.randrange(len(dropout))
    filter_idx = random.randrange(len(frame_filter_size))
    hid_idx = random.randrange(len(hidden_size))
    btl_idx = random.randrange(len(bottleneck_feats))
    
    expt_call = (
        f"{base_call} "
        f"-lr {learning_rates[lr_idx]} "
        f"-wd {weight_decays[wd_idx]} "
        f"-flat {flatten_methods[fl_idx]} "
        f"-l {lstm_layers[lstm_idx]} "
        f"-cnn {cnn_layers[cnn_idx]} "
        f"-dr {dropout[dr_idx]} "
        f"-f {frame_filter_size[filter_idx]} "
        f"-pad {frame_pad_size[filter_idx]} "
        f"-hid {hidden_size[hid_idx]} "
        f"-b {bottleneck_feats[btl_idx]}"
    )
    print(expt_call, file=output_file)

output_file.close()
