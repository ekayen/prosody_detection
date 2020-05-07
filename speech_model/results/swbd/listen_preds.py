import os
import pandas
import pickle
import sys 
from itertools import islice
import subprocess

pred_file = sys.argv[1]
swbd_dict = '../../../data/swbd_new_only/swbd_new_only.pkl'
swbd_aud_dir = '/group/corporapublic/switchboard/switchboard1/swb1'

with open(swbd_dict,'rb') as f:
    swbd = pickle.load(f)

N = 14
with open(pred_file, 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for i in range(int(len(lines)/N)):
        ex = lines[i*N:(i*N)+N]
        utt_id = ex[0]
        conv_id = utt_id.split('_')[0]
        aud_file = 'sw0'+conv_id[-4:]+'.sph'
        start= swbd['utt2toktimes'][utt_id][0]
        end= swbd['utt2toktimes'][utt_id][-1]
        to_print = '\n'.join(ex[1:])
        subprocess.check_output(['play',os.path.join(swbd_aud_dir,aud_file),'trim',start,f'={end}'])
        import pdb;pdb.set_trace()
