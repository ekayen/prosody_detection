"""
Read in the feats.scp file created by Kaldi feature extraction and
create a python dict and pickle it.

Only keep feats for those utterances that are in the train/dev/test data (swbd nxt corpus)

"""
import kaldi_io
import pandas as pd
import pickle
import torch

#feats_file = '/home/elizabeth/repos/kaldi/egs/swbd/s5c/data/train/feats_pitch.scp'
feats_file = '/afs/inf.ed.ac.uk/group/project/prosody/mfcc_pitch/feats.scp'
data_file = 'data/utterances.txt'
feat_dict_file = 'data/utterances_feats.pkl'
label_dict_file = 'data/utterances_labels.pkl'

df = pd.read_csv(data_file,sep='\t')
labels = df.iloc[:,-1].tolist()
keepkeys = df.iloc[:,1].tolist()

label_dict = {}
for i,key in enumerate(keepkeys):
    label_dict[key] = torch.tensor(int(labels[i].split()[-1]))


feat_dict = {}
print("filtering keys ...")
for key,mat in kaldi_io.read_mat_scp(feats_file):
    if key in keepkeys:
        print(key)
        feat_dict[key] = torch.tensor(mat)


with open(feat_dict_file,'wb') as f:
    pickle.dump(feat_dict,f)
with open(label_dict_file,'wb') as f:
    pickle.dump(label_dict,f)
