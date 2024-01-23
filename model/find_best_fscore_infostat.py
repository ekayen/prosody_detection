import os
import pandas as pd
import numpy as np

"""
Iterate through all result tsvs in a directory and find their best fscore performance.
"""

outdir = 'results/swbd/thesis'

#best_df = pd.DataFrame(columns=["num_classes","old_vs_new","eval","old_f_score","new_f_score"])

best_dict = {'idx':[],
             'filename':[],
             'num_classes':[],
             'old_vs_new':[],
             'eval':[],
             'old_f_score':[],
             'new_f_score':[],
             'input':[]}

f_score_name = 'dev_np_f'


idx = 0
for filename in os.listdir(outdir):
    if filename.endswith('tsv'):
        best_dict['idx'].append(idx)
        best_dict['filename'].append(filename)
        df = pd.read_csv(os.path.join(outdir,filename),sep='\t')
        fscores = [[float(i) for i in f.strip('[').strip(']').split()] for f in df[f_score_name]]
        if '_old' in filename:
            old_idx = 1
            new_idx = 0
            best_dict['old_vs_new'].append('old')
        else:
            old_idx = 0
            new_idx = 1
            best_dict['old_vs_new'].append('new')
        old_fscores = [f[old_idx] for f in fscores]
        new_fscores = [f[new_idx] for f in fscores]
        if 'old_np' or 'new_np' in filename:
            best_dict['num_classes'].append(3)
        else:
            best_dict['num_classes'].append(2)
        if 'eval_only_np' in filename:
            best_dict['eval'].append('np')
        else:
            best_dict['eval'].append('all')
        if 'text' in filename:
            best_dict['input'].append('text')
        elif 'speech' in filename:
            best_dict['input'].append('speech')
        elif 'both' in filename:
            best_dict['input'].append('both')
        else:
            best_dict['input'].append('speech')
        best_dict['old_f_score'].append(max(old_fscores))
        best_dict['new_f_score'].append(max(new_fscores))
        idx += 1
best_df = pd.DataFrame.from_dict(best_dict)
import pdb;pdb.set_trace()
