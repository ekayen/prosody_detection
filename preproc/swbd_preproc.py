import re
import os
from string import punctuation
import nltk
import kaldi_io
import pickle
nltk.download('punkt')
import pandas as pd
import torch


class BurncPreprocessor:
    def __init__(self,swbd_dir,pros_feat_dir,save_dir):

        self.swbd_dir = swbd_dir
        self.save_dir = save_dir
        self.pros_feat_dir = pros_feat_dir

        self.pros_feat_names = ['audspec_lengthL1norm_sma',
                                'voicingFinalUnclipped_sma',
                                'pcm_RMSenergy_sma',
                                'logHNR_sma',
                                'F0final_sma',
                                'pcm_zcr_sma']

        self.para_ids = []
        self.utterances = []
        self.utt2text = {}
        self.utt2spk = {}
        self.utt2recording = {}
        self.recording2file = {}
        self.utt2startend = {}  # store start and end timestamp of utterance
        self.utt2tokentimes = {}  # store start time of each token (not necessary for kaldi, but plan to use in model)
        self.utt2tones = {}
        self.utt_ids = []
        self.utt2frames = {}

        self.para2utt = {}
        self.utt2para = {}
        self.utt2toks = {}
        self.tok2utt = {}
        self.tok2tone = {}
        self.tok2times = {}
        self.tok2tokstr = {}
        self.tok2prosfeats = {}
        self.tok2mfccfeats = {}

    def gen_nested_dict(self):
        print('generating nested dict ...')
        """
        for para in self.para2utt:
            print(para)
            self.nested[para] = {
                'utterances': dict([(utt_id,self.utt2toks[utt_id]) for utt_id in self.para2utt[para]]),
                'tokens': [tok for utt_id in self.para2utt[para] for tok in self.utt2toks[utt_id]], # TODO maybe make a dict and add utt_ids as values?
                'mfccs': dict([(tok, self.tok2mfccfeats[tok]) for utt_id in self.para2utt[para] for tok in
                               self.utt2toks[utt_id]]),
                'prosfeats': dict([(tok, self.tok2prosfeats[tok]) for utt_id in self.para2utt[para] for tok in
                               self.utt2toks[utt_id]]),
                'tok2times':  dict([(tok,self.tok2times[tok]) for utt_id in self.para2utt[para] for tok in self.utt2toks[utt_id]]),
                'tok2tokstr': dict([(tok,self.tok2tokstr[tok]) for utt_id in self.para2utt[para] for tok in self.utt2toks[utt_id]]),
                'tok2tone': dict([(tok,self.tok2tone[tok]) for utt_id in self.para2utt[para] for tok in self.utt2toks[utt_id]]),
                'tok2utt': dict([(tok,self.tok2utt[tok]) for utt_id in self.para2utt[para] for tok in self.utt2toks[utt_id]]),
            }
        """
        # DECISION: use sentences as designated in syntax file.
        self.nested['utt2toks'] = self.utt2toks
        self.nested['tok2pros'] = self.tok2prosfeats
        self.nested['tok2str'] = self.tok2tokstr
        self.nested['tok2times'] = self.tok2times
        self.nested['tok2tone'] = self.tok2tone
        self.nested['utt_ids'] = self.utt_ids
        self.nested['tok2utt'] = self.tok2utt
        self.nested['utt2frames'] = self.utt2frames

        # Go through things in annotated_files.txt in the pros_feat_dir
        #   Find corresponding syntax file
        #   Go through each sentence in syntax file (store in utt_ids)
        #      Go through each terminal in sentence
        #      Grab feats
        #      store feats in tok2prosfeats
        #      store tok in utt2toks
        #      store in tok2utt
        #      store terminal in tok2str
        #      also store toktimes in tok2toktimes, and utt2frames




def main():
    swbd_dir = '/group/corporapublic/switchboard/nxt'
    pros_feat_dir = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/swbd'

