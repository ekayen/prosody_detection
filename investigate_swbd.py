import os
import pickle


data_path = '/afs/inf.ed.ac.uk/group/project/prosody/prosody_detection/data/swbd/swbd.pkl'
print('loading...')
swbd_dict = pickle.load(open(data_path,'rb'))
print('done.')


utt2toks = swbd_dict['utt2toks']
tok2tone = swbd_dict['tok2tone']

ones = 0
zeros = 0


for utt in utt2toks:
    for tok in utt2toks[utt]:
        if  tok in tok2tone:
            tone = tok2tone[tok]

            if tone == 0:
                zeros += 1
            if tone == 1:
                ones += 1

#assert ones+zeros == len(tok2tone)

import pdb;pdb.set_trace()
