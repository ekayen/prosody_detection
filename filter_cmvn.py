#!/usr/bin/env python
# coding: utf-8
"""
Loads CMVN-ed features, which are in a text file, not a binary file.
Takes ~1 gazillion years to run.
Should be done in one stage, but the commented part was writing the features as strings,
not tensors of floats, so I redid the dictionary in the uncommented part,
rather than redoing the first part altogether (cf. gazillion years).

"""

import numpy as np
import pickle
import pandas as pd
import psutil
import os



data_file = '/afs/inf.ed.ac.uk/group/project/prosody/stars/data/utterances.txt'

df = pd.read_csv(data_file,sep='\t',header=None)
keepkeys = set(df.iloc[:,1].tolist())



filepath = '/afs/inf.ed.ac.uk/group/project/prosody/mfcc_pitch/all_cmvn.ark'

cmvn_dict = {}

key = ''
tmp = []
counter = 0
"""
# Load features into a dictionary. Accidentally make empty dictionary entries for 
# the keys you meant to remove. Also accidentally write the features as strings rather than tensors of float32
with open(filepath,'r') as f:
    for line in f:
        print('key:',key)
        if '[' in line:
            if key:
                cmvn_dict[key] = tmp
                tmp = []
            key = line.split()[0].strip()
        if key in keepkeys:
            if ']' in line:
                line = line.strip().rstrip(']').strip()
                tmp.append(line)
            else:
                tmp.append(line)
                
        process = psutil.Process(os.getpid())
        print('Memory usage:',process.memory_info().rss/1000000000, 'GB')
        counter += 1
        
"""

# Having screwed up the first part, reload the dictionary,
# dump the irrelevant keys, and write the features as tensors of float32

with open('cmvn_dict.pkl','wb') as f:
    pickle.dump(cmvn_dict,f)


import torch
import numpy as np
with open('data/cmvn_dict.pkl','rb') as f:
    cmvn_dict = pickle.load(f)

out_dict = {}
for key in cmvn_dict:
    if cmvn_dict[key]:
        str_arr = cmvn_dict[key][1:]
        arr = np.array([[float(entry) for entry in line.strip().split()] for line in str_arr])
        out_dict[key] = torch.tensor(arr,dtype=torch.float32)
    else:
        print('empty at',key)



with open('cmvn_tensors.pkl','wb') as f:
    pickle.dump(out_dict,f)


with open('cmvn_tensors.pkl','rb') as f:
    check_dict = pickle.load(f)

keptkeys = set(check_dict.keys())

print(keptkeys-keepkeys)
print(keepkeys-keptkeys)