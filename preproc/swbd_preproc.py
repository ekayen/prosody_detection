import re
import os
from string import punctuation
import nltk
import kaldi_io
import pickle
nltk.download('punkt')
import pandas as pd
import torch
from bs4 import BeautifulSoup

class SwbdPreprocessor:
    def __init__(self,swbd_dir,pros_feat_dir,save_dir,
                 annotated_files='/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/swbd/annotated_files.txt'):

        self.swbd_dir = swbd_dir
        self.save_dir = save_dir
        self.pros_feat_dir = pros_feat_dir
        self.annotated_files = annotated_files

        self.pros_feat_names = ['audspec_lengthL1norm_sma',
                                'voicingFinalUnclipped_sma',
                                'pcm_RMSenergy_sma',
                                'logHNR_sma',
                                'F0final_sma',
                                'pcm_zcr_sma']

        self.utterances = []
        self.utt2text = {}
        self.utt2spk = {}
        self.utt2recording = {}
        self.utt2startend = {}  # store start and end timestamp of utterance
        self.utt2tokentimes = {}  # store start time of each token (not necessary for kaldi, but plan to use in model)
        self.utt2tones = {}
        self.utt_ids = []
        self.utt2frames = {}

        self.utt2toks = {}
        self.tok2utt = {}
        self.tok2tone = {}
        self.tok2times = {}
        self.tok2tokstr = {}
        self.tok2prosfeats = {}
        self.tok2mfccfeats = {}
        self.tok2infostat = {}

    def get_pros_feats(self):
        pass

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
        #
        #      utt ids = sw<conversation ID>s<sentence number>
        #      tok ids = terminal ids in the corpus: sw<conversation_id>s<sentence_number>_<token number>

    def text_preproc(self,file_list):
        '''
        Create the following:
        self.utterances
        self.utt2text
        self.utt2spk
        self.utt2startend
        self.utt2tokentimes
        self.utt2tones
        self.utt_ids
        self.utt2frames
        self.utt2toks
        self.tok2utt
        self.tok2tone
        self.tok2times
        self.tok2tokstr
        self.tok2infostat
        '''
        for file in file_list:
            conversation,speaker,_,_ = file.strip().split('.')
            syntax_path = os.path.join(self.swbd_dir, 'syntax', '.'.join([conversation,speaker,'syntax','xml']))
            term_path = os.path.join(self.swbd_dir, 'terminals', '.'.join([conversation,speaker,'terminals','xml']))
            syn_file = open(syntax_path, "r")
            syn_contents = syn_file.read()
            syn_soup = BeautifulSoup(syn_contents, 'lxml')
            sentences = syn_soup.find_all('parse') # TODO paused here with the sentences loaded but unprocessed.
            # TODO Next thing to do is go through them, generate the utt ids, the tok ids, and all the dicts that
            # TODO are possible to build a this point
            import pdb;pdb.set_trace()

            #term_file = open(term_path, "r")
            #term_contents = term_file.read()
            #term_soup = BeautifulSoup(term_contents, 'lxml')
            #import pdb;pdb.set_trace()



    def load_opensmile_feats(self):
        pass

    def acoustic_preproc(self):
        self.load_opensmile_feats()


    def preproc(self,write_dict=True,out_file='swbd.pkl'):
        with open(self.annotated_files,'r') as f:
            file_list = f.readlines()
        self.text_preproc(file_list)
        self.acoustic_preproc()
        self.gen_nested_dict()
        if write_dict:
            self.save_nested(name=out_file)



def main():
    swbd_dir = '/group/corporapublic/switchboard/nxt/xml'
    pros_feat_dir = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/swbd'
    save_dir = '../data/swbd'
    annotated_files = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/swbd/annotated_files.txt'
    preprocessor = SwbdPreprocessor(swbd_dir,pros_feat_dir,save_dir,annotated_files)
    preprocessor.preproc()

if __name__=="__main__":
    main()

