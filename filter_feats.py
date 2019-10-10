"""
Read in the feats.scp file created by Kaldi feature extraction and
create a python dict and pickle it.

Only keep feats for those utterances that are in the train/dev/test data (swbd nxt corpus)

"""
import kaldi_io
import pandas as pd
import pickle

feats_file = '/home/elizabeth/repos/kaldi/egs/swbd/s5c/data/train/feats_pitch.scp'
data_file = 'data/utterances.txt'
feat_dict_file = 'data/utterances_feats.pkl'

df = pd.read_csv(data_file,sep='\t')
keepkeys = set(df.iloc[:,1].tolist())

feat_dict = {}
print("filtering keys ...")
for key,mat in kaldi_io.read_mat_scp(feats_file):

    if key in keepkeys:
        print(key)
        feat_dict[key] = mat

with open(feat_dict_file,'wb') as f:
    pickle.dump(feat_dict,f)
