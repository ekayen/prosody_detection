import pickle
import torch
import numpy as np

#with open('../data/swbd/swbd_acc.pkl','rb') as f:
with open('../data/swbd/swbd.pkl','rb') as f:
    swbd_dict = pickle.load(f)

THRESHOLD = .25

gaps = 0
gap_sizes = []
gap_toks = {}
gap_texts = {}

for utt in swbd_dict['utt_ids']:
    toks = swbd_dict['utt2toks'][utt]
    for i in range(len(toks)-1):
        tok1 = toks[i]
        tok2 = toks[i+1]
        end1 = swbd_dict['tok2times'][tok1][-1]
        start2 = swbd_dict['tok2times'][tok2][0]
        if not end1 == start2:
            gap = start2-end1
            if gap > THRESHOLD:
                gaps += 1

                gap_sizes.append(gap)
                gap_toks[gap] = (tok1,tok2)
                utt = swbd_dict['tok2utt'][tok1]
                text = [swbd_dict['tok2str'][tok] for tok in swbd_dict['utt2toks'][utt]]
                gap_texts[gap] = text

empty_utts = []
for utt in swbd_dict['utt2toks']:
    if len(swbd_dict['utt2toks'][utt])==1 and swbd_dict['utt2frames'][utt][1]-swbd_dict['utt2frames'][utt][0]:
        print('empty utt')
        empty_utts.append(utt)


import pdb;pdb.set_trace()

#TODO probably gotta deal with this by splitting at the gaps and just having more utterances
