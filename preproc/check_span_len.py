import pickle
import torch



with open('../data/burnc/span2feat_open.pkl','rb') as f:
    span2feat = pickle.load(f)

seq_lens = []

for span in span2feat:
    print(span2feat[span].shape)
    seq_lens.append(span2feat[span].shape[0])

import pdb;pdb.set_trace()