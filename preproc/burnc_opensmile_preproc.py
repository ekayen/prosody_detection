import os
import pandas as pd
import torch

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

#df = pd.read_csv(os.path.join(data_path,'tmp.csv'),sep=';')



for key in keys:
    print(key)
    file = key2files[key]
    range = key2range[key]
    df = pd.read_csv(os.path.join(data_path,file),sep=';')
    print(df.columns)
    df = df[['frameIndex',
             'frameTime',
             'audspec_lengthL1norm_sma',
             'voicingFinalUnclipped_sma.1',
             'pcm_RMSenergy_sma',
             'HarmonicsToNoiseRatioACFLogdB_sma',
             'F0final_sma.1',
             'pcm_zcr_sma']]

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
    for row in utt_df.iterrows():
        tens = torch.tensor(row.tolist())
        feat_tensors.append(tens)


    import pdb;pdb.set_trace()

