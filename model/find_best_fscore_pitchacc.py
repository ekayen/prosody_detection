import os
import pandas as pd
import numpy as np

"""
Iterate through all result tsvs in a directory and find their best fscore performance.
"""

outdir = 'results/thesis'

#best_df = pd.DataFrame(columns=["num_classes","old_vs_new","eval","old_f_score","new_f_score"])

best_dict = {'idx':[],
             'filename':[],
             'acc':[],
             'f_score_0':[],
             'f_score_1':[],
             'best_acc_epoch':[],
             'best_f_0_epoch':[],
             'best_f_1_epoch':[],
             'input':[]}

f_score_name = 'dev_f'
acc_name = 'dev_accs'

idx = 0
for filename in os.listdir(outdir):
    if filename.endswith('tsv') and not 'best' in filename:
        best_dict['idx'].append(idx)
        best_dict['filename'].append(filename)
        df = pd.read_csv(os.path.join(outdir,filename),sep='\t')

        fscores = [[float(i) for i in f.strip('[').strip(']').split()] for f in df[f_score_name]]
        accs = np.array([float(i) for i in df[acc_name]])
        
        f_scores_0 = np.array([f[0] for f in fscores])
        f_scores_1 = np.array([f[1] for f in fscores])
        if 'text' in filename:
            best_dict['input'].append('text')
        elif 'speech' in filename:
            best_dict['input'].append('speech')
        elif 'both' in filename:
            best_dict['input'].append('both')
        else:
            best_dict['input'].append('speech')
        best_dict['f_score_0'].append(max(f_scores_0)) # TODO, wait, this means that I'll end up picking the best, even if its a different model. Hm. Want to keep track of which model was best.
        best_dict['f_score_1'].append(max(f_scores_1))
        best_dict['acc'].append(max(accs))
        best_dict['best_acc_epoch'].append(np.argmax(accs))
        best_dict['best_f_0_epoch'].append(np.argmax(f_scores_0))
        best_dict['best_f_1_epoch'].append(np.argmax(f_scores_1))
        idx += 1
best_df = pd.DataFrame.from_dict(best_dict)
swbd_df = best_df[best_df["filename"].str.contains("swbd")]
burnc_df = best_df[~best_df["filename"].str.contains("swbd")]
best_df.to_csv(os.path.join(outdir,'best.tsv'),sep='\t')
import pdb;pdb.set_trace()
