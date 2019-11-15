import os
import pandas as pd
import torch
import pickle

#### STEP 1: Find utterance keys and find what files and frames they correspond to
burnc_keys_path = '../data/burnc/text2labels_breath_tok'
data_path = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/burnc'

keys = pd.read_csv(burnc_keys_path,sep='\t',header=None)[0].tolist()
ranges = [(float(key.split('-')[1]),float(key.split('-')[2])) for key in keys]
files = [key.split('-')[0]+'.csv' for key in keys]

key2range = dict(zip(keys,ranges))
key2files = dict(zip(keys,files))

#### STEP 2: for each key, open up the correct feat file and load the feats as a df
#### Then drop the unused columns, make each row into an array, keyed by the frame,
#### and then select the right frames for your utterance

key2feats = {}
for key in keys:
    print(key)
    file = key2files[key]
    range = key2range[key]
    orig_df = pd.read_csv(os.path.join(data_path,file),sep=';')
    #print(df.columns)
    df = orig_df[['frameIndex',
             'frameTime',
             'audspec_lengthL1norm_sma',
             'voicingFinalUnclipped_sma.1',
             'pcm_RMSenergy_sma',
             'HarmonicsToNoiseRatioACFLogdB_sma',
             'F0final_sma.1',
             'pcm_zcr_sma']]

    import pdb;pdb.set_trace()

    #utt_df = df.loc[df['frameTime'] >= range[0] and df['frameTime'] <= range[1]]
    utt_df = df.loc[df['frameTime'] >= range[0]]
    utt_df = utt_df.loc[utt_df['frameTime'] <= range[1]]
    utt_df = utt_df[['audspec_lengthL1norm_sma',
                     'voicingFinalUnclipped_sma.1',
                     'pcm_RMSenergy_sma',
                     'HarmonicsToNoiseRatioACFLogdB_sma',
                     'F0final_sma.1',
                     'pcm_zcr_sma']]

    feat_tensors = []
    for i,row in utt_df.iterrows():
        tens = torch.tensor(row.tolist()).view(1,len(row.tolist()))
        feat_tensors.append(tens)

    key2feats[key] = torch.cat(feat_tensors,dim=0)

with open('../data/burnc/burnc_mfcc_dict.pkl','rb') as f:
    mfcc_dict = pickle.load(f)





with open('../data/burnc/burnc_breath_opensmilefeats.pkl','wb') as f:
    pickle.dump(key2feats,f)