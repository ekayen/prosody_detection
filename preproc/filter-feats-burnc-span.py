#!/usr/bin/env python
# coding: utf-8

#NOTE Have to throw out f2bs02 because the tokens are misaligned in the annotation.
BROKEN_ANNO = ['f2bs02p1']

import os
import pandas as pd
import pickle
import torch
import kaldi_io
import torch
import numpy as np

bitmark = False
FEAT_SOURCE = 'opensmile'
#FEAT_SOURCE = 'kaldi'

outdir = '../data/burnc'


# ### **Step 1:** Load in the info about the spans

# Load some necessary stuff:

spanfile = 'burnc_breath/spans'
span_df = pd.read_csv(spanfile,sep='\t',header=None)
lbl_file = '../data/burnc/text2labels_breath_tok'

# Extract info that you'll use for both kaldi and opensmile

para_ids = span_df[0].tolist()
toks = span_df[1].tolist()
lbls = span_df[2].tolist()
toktimes = span_df[3].tolist()
timespans = [(int(round(float(tims.split()[0])*100)),int(round(float(tims.split()[-1])*100))) for tims in toktimes]

toktimes = [[int(round(float(t)*100)) for t in tims.split()] for tims in toktimes]

df = pd.read_csv(lbl_file,sep='\t',header=None)
utt_ids = df[0].tolist()



# Load in the kaldi-generated features:
# Using the breath tokenization from Kaldi -- only matters because the last two frames of each utterance are truncated -- shouldn't make a huge difference in general

if FEAT_SOURCE=='kaldi':
    feats_file = '/home/elizabeth/repos/kaldi/egs/burnc/kaldi_features/data/train_breath_tok/feats.scp'



    feat_dict = {}
    for utt_id,mat in kaldi_io.read_mat_scp(feats_file):
        if utt_id in utt_ids:
            feat_dict[utt_id] = mat


    # Try to make a nested dictionary that is keyed on 1) recording ID and 2) frame number. That way I can pick out the exact frames I want for my spans

    frame_dict = {}
    for utt in feat_dict:
        para_id,start,end = utt.split('-')
        start = int(round(float(start)*100))
        end = int(round(float(end)*100))
        frames = feat_dict[utt]
        for i in range(frames.shape[0]):
            if not para_id in frame_dict:
                frame_dict[para_id] = {}
            frame_dict[para_id][start+(i)] = frames[i,:]
        


    # Next go through each span and create a tensor for the audio features (either bitmarked or not) and save that and everything else you need to separate dictionaries



    span2feat = {}
    span2tok = {}
    span2toktimes = {}
    span2lbl = {}
    for i,timespan in enumerate(timespans):
        start,end = timespan
        para_id = para_ids[i]
        #print(para_id)
        tok = toks[i]
        lbl = lbls[i]
        toktime = toktimes[i]
        foc_start,foc_end = toktime[1],toktime[2]

        frame_idx = start
        feats = []
        while frame_idx <= end:
            try:
                feat_vec = frame_dict[para_id][frame_idx]
                if bitmark:
                    featfile = 'span2feat_bitmark.pkl'
                    if frame_idx > foc_start and frame_idx < foc_end:
                        feat_vec = np.append(feat_vec,[1])
                    else:
                        feat_vec = np.append(feat_vec,[0])
                else:
                    featfile = 'span2feat.pkl'
                #print(feat_vec.shape)
                feats.append(feat_vec)

            except KeyError:
                #print('Didnt find key ',frame_idx,'in',para_id)
                pass
            frame_idx += 1

        feats = [np.expand_dims(ft,axis=0) for ft in feats]
        if len(feats) > 0:
            feats = np.concatenate(feats,axis=0)

            tens = torch.tensor(feats,dtype=torch.float32)
            span_id = para_id+'-'+'%08.3f'%start+'-'+'%08.3f'%end
            span2feat[span_id] = tens
            span2lbl[span_id] = lbl
            span2tok[span_id] = tok
            span2toktimes[span_id] = toktime
        else:
            print('dropped',para_id,start,end)

    tokfile = 'span2tok.pkl'
    lblfile = 'span2lbl.pkl'
    toktimefile = 'span2toktimes.pkl'


# ### **Step 2:** Process the spans, using OpenSmile features
elif FEAT_SOURCE=='opensmile':

    datadir = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/burnc'

    ranges = [(float(key.split('-')[1]),float(key.split('-')[2])) for key in utt_ids]
    files = [key.split('-')[0]+'.csv' for key in utt_ids]

    import pdb;pdb.set_trace()

    span2feat = {}
    span2tok = {}
    span2toktimes = {}
    span2lbl = {}
    idx = 0
    for i,timespan in enumerate(timespans):
        start,end = timespan
        para_id = para_ids[i]
        if not para_id in BROKEN_ANNO:
            span_id = para_id+'-'+'%08.3f'%start+'-'+'%08.3f'%end
            print(span_id)
            tok = toks[i]
            lbl = lbls[i]
            toktime = toktimes[i]
            foc_start,foc_end = toktime[1],toktime[2]
            feat_file = os.path.join(datadir,para_id+'.is13.csv')
            feat_df = pd.read_csv(feat_file,sep=';')
            feat_df = feat_df[['frameTime',
                               'audspec_lengthL1norm_sma',
                               'voicingFinalUnclipped_sma',
                               'pcm_RMSenergy_sma',
                               'logHNR_sma',
                               'F0final_sma',
                               'pcm_zcr_sma']]

            feat_df['frameIndex'] = feat_df['frameTime'] * 100
            feat_df['frameIndex'] = feat_df['frameIndex'].astype(int)



            span_df = feat_df.loc[feat_df['frameIndex'] >= start]
            span_df = span_df.loc[span_df['frameIndex'] <= end]


            if bitmark:
                featfile = 'span2feat_bitmark_open.pkl'
                span_df['bitmark'] = 0
                tok_brk_1 = toktime[1]
                tok_brk_2 = toktime[2]
                span_df.loc[span_df['frameIndex'] >= tok_brk_1, 'bitmark'] = 1
                span_df.loc[span_df['frameIndex'] > tok_brk_2, 'bitmark'] = 0

                span_df = span_df[['audspec_lengthL1norm_sma',
                                   'voicingFinalUnclipped_sma',
                                   'pcm_RMSenergy_sma',
                                   'logHNR_sma',
                                   'F0final_sma',
                                   'pcm_zcr_sma',
                                   'bitmark']]
            else:
                featfile = 'span2feat_open.pkl'
                span_df = span_df[['audspec_lengthL1norm_sma',
                                   'voicingFinalUnclipped_sma',
                                   'pcm_RMSenergy_sma',
                                   'logHNR_sma',
                                   'F0final_sma',
                                   'pcm_zcr_sma']]



            feat_tensors = []
            for i,row in span_df.iterrows():
                tens = torch.tensor(row.tolist()).view(1,len(row.tolist()))
                feat_tensors.append(tens)

            if not feat_tensors:
                import pdb;pdb.set_trace()
            span2feat[span_id] = torch.cat(feat_tensors,dim=0)
            span2lbl[span_id] = lbl
            span2tok[span_id] = tok
            span2toktimes[span_id] = toktime
    tokfile = 'span2tok_open.pkl'
    lblfile = 'span2lbl_open.pkl'
    toktimefile = 'span2toktimes_open.pkl'

with open(os.path.join(outdir,featfile),'wb') as f:
    pickle.dump(span2feat,f)
with open(os.path.join(outdir,tokfile),'wb') as f:
    pickle.dump(span2tok,f)
with open(os.path.join(outdir,lblfile),'wb') as f:
    pickle.dump(span2lbl,f)
with open(os.path.join(outdir,toktimefile),'wb') as f:
    pickle.dump(span2toktimes,f)





