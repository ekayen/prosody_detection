"""
Prepare BU radio news corpus data for kaldi feat extraction

Modeled on Herman Kamper's recipe for Mboshi parallel data
"""
import re
import os
from string import punctuation
import nltk
import kaldi_io
import pickle
nltk.download('punkt')
import pandas as pd
import torch
import yaml
import json


TOKENIZATION_METHOD = 'breath_tok'

class MustcPreprocessor:
    def __init__(self,mustc_dir,pros_feat_dir,save_dir,split='dev'):

        self.mustc_dir = mustc_dir
        self.save_dir = save_dir
        self.pros_feat_dir = pros_feat_dir
        self.split = split


        self.pros_feat_names = ['audspec_lengthL1norm_sma',
                                'voicingFinalUnclipped_sma',
                                'pcm_RMSenergy_sma',
                                'logHNR_sma',
                                'F0final_sma',
                                'pcm_zcr_sma']

        self.para_ids = []
        self.three_tok_spans = []
        self.speakers_used = set()
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
        self.utt2prosfeats = {}

        self.para2utt = {}
        self.utt2para = {}
        self.utt2toks = {}
        self.tok2utt = {}
        self.tok2tone = {}
        self.tok2times = {}
        self.tok2tokstr = {}
        self.tok2prosfeats = {}

        # Final nested dict of all features
        self.nested = {}

        self.alignments = []

    @staticmethod
    def text_reg(word):
        remove = punctuation.replace('-','').replace('<','').replace('>','')
        #remove = punctuation.replace('<', '').replace('>', '')
        word = word.lower().replace("'s","").replace("n't","").replace('/n','').replace('/v','')
        word = word.translate(str.maketrans('', '', remove))
        word = word.replace('-',' ')
        word = word.replace('—',' ')
        return word

    @staticmethod
    def matching(tok, word):
        tok = tok.lower().replace("'", "")
        word = word.lower().replace("'", "")
        if tok==word:
            return True
        elif word.rstrip("s")==tok:
            return True
        elif word.rstrip("t").rstrip("n")==tok:
            return True
        elif word == 'donts' and tok == 'dos':
            return True
        else:
            return False

    @staticmethod
    def convert_to_sec(timestamp):
        h, m, s = timestamp.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)


    def process_utt(self,words,timestamps,para_id,spkr,recordingid,tones_per_word,idx):
        pass

    def load_alignments(self):
        align_files = []
        for filename in os.listdir(os.path.join(self.mustc_dir, 'wav')):
            if filename.endswith('.align'):
                align_files.append(filename)
        align_files = sorted(align_files)
        all_words = []
        for filename in align_files:
            with open(os.path.join(self.mustc_dir, 'wav', filename), 'r') as f:
                json_dict = f.read()
                align_dict = json.loads(json_dict)
                words = align_dict['words'] # TODO order this bit (may not work)
                all_words.extend(words)
        self.alignments = all_words

    def text_preproc(self):
        self.load_alignments()
        with open(os.path.join(self.mustc_dir, 'txt', f'{self.split}.en'),'r') as f:
            en_lines = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.mustc_dir,'txt',f'{self.split}.yaml'),'r') as f:
            utt_list = yaml.load(f, Loader=yaml.FullLoader)

        null_toks = []

        num_aligned_toks = len(self.alignments)
        num_manual_toks = 0
        for line in en_lines:
            line = line.split()
            num_manual_toks += len(line)
        print(f'num_aligned_toks: {num_aligned_toks}')
        print(f'num_manual_toks: {num_manual_toks}')
        #import pdb;pdb.set_trace()

        tok_counter = 0
        for i in range(len(utt_list)):
            if tok_counter%1000 == 0:
                print(tok_counter)
            en = self.text_reg(en_lines[i])
            toks = en.split()
            utt_dict = utt_list[i]
            duration = utt_dict['duration']
            offset = utt_dict['offset']
            spk = utt_dict['speaker_id']
            utt_id = f'{spk}_{i}'
            wav = utt_dict['wav']
            self.utt_ids.append(utt_id)
            self.utt2text[utt_id] = en
            self.utt2spk[utt_id] = spk
            self.utt2recording[utt_id] = wav

            # Load actual prosodic features
            feat_file = os.path.join(self.pros_feat_dir, wav.replace('wav', 'is13.csv'))
            feat_df = pd.read_csv(feat_file, sep=';')

            tok_ids = []
            tok_times = []
            for tok in toks:
                if self.alignments[tok_counter]['case'] == 'success':
                    tok_id = f'{utt_id}_{tok_counter}'
                    self.tok2utt[tok_id] = utt_id
                    self.tok2tokstr[tok_id] = tok
                    self.tok2tone[tok_id] = 0
                    tok_ids.append(tok_id)
                    start = self.alignments[tok_counter]['start']
                    end = self.alignments[tok_counter]['end']
                    tok_times.append(start)
                    self.tok2times[tok_id] = (start,end)
                    tok_df = feat_df.loc[feat_df['frameTime'] >= start]
                    tok_df = tok_df.loc[tok_df['frameTime'] < end]
                    tok_df = tok_df[self.pros_feat_names]
                    feat_tensors = []
                    word = self.alignments[tok_counter]['word']
                    if not self.matching(tok,word):
                        print('------------')
                        print(tok)
                        print(word)
                        print('++++++++++++')
                        import pdb;pdb.set_trace()
                    for i, row in tok_df.iterrows():
                        tens = torch.tensor(row.tolist()).view(1, len(row.tolist()))
                        feat_tensors.append(tens)
                    try:
                        tok_feat_tensors = torch.cat(feat_tensors, dim=0)
                        self.tok2prosfeats[tok_id] = tok_feat_tensors
                    except:
                        import pdb;pdb.set_trace()
                        null_toks.append((tok_id,tok_df))
                    tok_counter += 1
                else:
                    #print(self.alignments[tok_counter])
                    tok_counter += 1
            tok_times.append(end)
            self.utt2toks[utt_id] = tok_ids
            self.utt2tokentimes[utt_id] = tok_times
            self.utt2frames[utt_id] = torch.tensor([int(round(100*(tim-tok_times[0]))) for tim in tok_times],dtype=torch.float32)
            #import pdb;pdb.set_trace()


        assert tok_counter == len(self.alignments)
        #import pdb;pdb.set_trace()






    def load_opensmile_feats(self):
        pass

    def trim_empty_utts(self):
        empty_utts = ['spk.837_1260'] #
        for utt in self.utt2toks:
            if self.utt2toks[utt] == []:
                empty_utts.append(utt)
        for utt in empty_utts:
            del self.utt2toks[utt]


    def gen_nested_dict(self):
        print('generating nested dict ...')

        self.trim_empty_utts()

        self.nested['utt2toks'] = self.utt2toks
        self.nested['tok2pros'] = self.tok2prosfeats
        self.nested['tok2str'] = self.tok2tokstr
        self.nested['tok2times'] = self.tok2times
        self.nested['tok2tone'] = self.tok2tone
        self.nested['utt2para'] = self.utt2para
        self.nested['utt_ids'] = self.utt_ids
        self.nested['tok2utt'] = self.tok2utt
        self.nested['para2utt'] = self.para2utt
        self.nested['utt2frames'] = self.utt2frames

    def save_nested(self,save_dir=None,name='mustc/mustc.pkl'):
        if not save_dir: save_dir = self.save_dir
        with open(os.path.join(save_dir,name),'wb') as f:
            pickle.dump(self.nested,f)

    def preproc(self,write_dict=True,out_file='mustc/mustc.pkl'):
        self.text_preproc()
        import pdb;pdb.set_trace()
        #self.acoustic_preproc()
        self.gen_nested_dict()
        if write_dict:
            self.save_nested(name=out_file)


def main():
    mustc_dir = '/home/elizabeth/en-ru/data/dev/'
    pros_feat_dir = '/home/elizabeth/opensmile-2.3.0/mustc'
    #mfcc_dir = '~/repos/kaldi/egs/burnc/kaldi_features/data/train_breath_tok/feats.scp'
    kaldi_dir = 'tmp' # Obsolete, can point anywhere
    save_dir = '../data'

    proc = MustcPreprocessor(mustc_dir,pros_feat_dir,save_dir,split='dev')
    proc.preproc(out_file='mustc.pkl')


if __name__ == "__main__":
    main()
