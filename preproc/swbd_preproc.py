import os
from string import punctuation
import pickle
import torch
from bs4 import BeautifulSoup
import pandas as pd



class SwbdPreprocessor:
    def __init__(self,swbd_dir,pros_feat_dir,save_dir,
                 annotated_files='/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/swbd/annotated_files.txt'):

        self.gap_threshold = 0.25
        self.nested = {}
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

        self.conv2utt = {}
        self.utt2conv = {}
        self.tok2conv = {}
        self.conv2tok = {}

        self.utt2text = {}
        self.utt2spk = {}
        self.utt2recording = {}
        self.utt2startend = {}  # store start and end timestamp of utterance
        self.utt2tokentimes = {}  # store start time of each token (not necessary for kaldi, but plan to use in model)
        self.utt2tones = {}
        self.utt_ids = []
        self.utt2frames = {}

        self.utt2toks = {}
        self.tok2spk = {}
        self.tok2utt = {}
        self.tok2tone = {}
        self.tok2times = {}
        self.tok2tokstr = {}
        self.tok2infostat = {}

        self.tok2prosfeats = {}

        self.correction = {'sw2370_ms33B_pw107':140.302000}

        self.utt2bio = {}
        self.utt2new = {}
        self.utt2old = {}

        self.allowed_infostats = set(['old','med','new'])

        self.tok2pos = {}
        self.tok2kontrast = {}
        self.utt2kontrast = {}

    def get_pros_feats(self):
        pass

    def del_tok(self,tok):
        if tok in self.tok2times: del self.tok2times[tok]
        if tok in self.tok2infostat: del self.tok2infostat[tok]
        if tok in self.tok2tokstr: del self.tok2tokstr[tok]
        if tok in self.tok2spk: del self.tok2spk[tok]
        if tok in self.tok2utt: del self.tok2utt[tok]
        if tok in self.tok2conv: del self.tok2conv[tok]

    def del_utt(self,utt):
        if utt in self.utt2toks: del self.utt2toks[utt]
        if utt in self.utt2tokentimes: del self.utt2tokentimes[utt]
        if utt in self.utt2startend: del self.utt2startend[utt]
        if utt in self.utt2conv: del self.utt2conv[utt]
        if utt in self.utt2frames: del self.utt2frames[utt]
        if utt in self.utt2spk: del self.utt2spk[utt]
        if utt in self.utt_ids: self.utt_ids.remove(utt)

    def gen_nested_dict(self,acc_only=False,kontrast_only=False):
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

        if acc_only:
            all_tok = set(self.tok2utt.keys())
            acc_tok = set(self.tok2tone.keys())
            non_acc_tok = all_tok - acc_tok

            for tok in non_acc_tok:
                utt = self.tok2utt[tok]
                self.del_tok(tok)
                self.del_utt(utt)

        if kontrast_only:
            all_tok = set(self.tok2utt.keys())
            kontrast_tok = set(self.tok2kontrast.keys())
            non_kontrast_tok = all_tok - kontrast_tok

            for tok in non_kontrast_tok:
                utt = self.tok2utt[tok]
                self.del_tok(tok)
                self.del_utt(utt)


        # Weed out utterances with no speech feats:
        del_utts = []
        del_toks = []
        for utt in self.utt2toks:
            if len(self.utt2toks[utt])==1:# and self.utt2tokentimes[utt][1]-self.utt2tokentimes[utt][1]:
                tok = self.utt2toks[utt][0]
                del_utts.append(utt)
                del_toks.append(tok)

        for utt in del_utts:
            self.del_utt(utt)
        for tok in del_toks:
            self.del_tok(tok)


        self.nested['conv2utt'] = self.conv2utt
        self.nested['utt2conv'] = self.utt2conv
        self.nested['utt2toktimes'] = self.utt2tokentimes
        self.nested['utt2spk'] = self.utt2spk
        self.nested['tok2infostat'] = self.tok2infostat
        self.nested['tok2spk'] = self.tok2spk
        self.nested['tok2conv'] = self.tok2conv

        self.nested['utt2toks'] = self.utt2toks
        self.nested['tok2pros'] = self.tok2prosfeats
        self.nested['tok2str'] = self.tok2tokstr
        self.nested['tok2times'] = self.tok2times
        self.nested['tok2tone'] = self.tok2tone
        self.nested['utt_ids'] = self.utt_ids
        self.nested['tok2utt'] = self.tok2utt
        self.nested['utt2frames'] = self.utt2frames
        self.nested['utt2bio'] = self.utt2bio
        self.nested['utt2new'] = self.utt2new
        self.nested['utt2old'] = self.utt2old
        self.nested['tok2pos'] = self.tok2pos
        self.nested['tok2kontrast'] = self.tok2kontrast
        self.nested['utt2kontrast'] = self.utt2kontrast

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
        #      utt ids = sw<conversation ID>_s<sentence number>
        #      tok ids = terminal ids in the corpus: sw<conversation_id>_s<sentence_number>_<token number>

    def save_nested(self,save_dir=None,name='swbd.pkl'):
        if not save_dir: save_dir = self.save_dir
        with open(os.path.join(save_dir,name),'wb') as f:
            pickle.dump(self.nested,f)

    @staticmethod
    def text_reg(word):
        remove = punctuation.replace('-','').replace('<','').replace('>','')
        word = word.lower().replace("'s","").replace("n't","").replace('/n','').replace('/v','')
        word = word.translate(str.maketrans('', '', remove))
        return word

    @staticmethod
    def get_id(conversation,num):
        return '_'.join([conversation,num])

    @staticmethod
    def extract_id_from_href(href):
        return href.split('(')[1].split(')')[0]

    def increment_id(self,id):
        id_elements = id.split('_')
        id_num = int(id_elements.pop())
        id_next = id_num + 1
        id_elements.append(str(id_next))
        return '_'.join(id_elements)

    def decrement_id(self,id):
        id_elements = id.split('_')
        id_num = int(id_elements.pop())
        id_next = id_num - 1
        id_elements.append(str(id_next))
        return '_'.join(id_elements)

    def utt_id_incr(self,utt_id):
        parts = utt_id.split('_')
        if len(parts) == 2:
            parts.append('a')
        elif len(parts) == 3:
            letter = parts.pop()
            letter = ord(letter[0])
            letter += 1
            letter = chr(letter)
            parts.append(letter)
        return '_'.join(parts)

    def text_preproc(self,file_list):
        '''
        Create the following:

        (can get from the sentence file alone)
        self.utt_ids
        self.utt2spk
        self.tok2spk
        self.tok2utt
        self.utt2toks

        (can get from terminal file alone)
        self.tok2times
        self.tok2tokstr

        (have to use info from both)
        self.utt2text
        self.utt2startend
        self.utt2tokentimes
        self.utt2frames

        (annotation files)
        self.tok2tone
        self.utt2tones
        self.tok2infostat
        '''
        pw2term = {}
        term2pw = {}


        for file in file_list:

            conversation,speaker,_,_ = file.strip().split('.')
            print(conversation)
            # Map from terminals to phonwords
            term_path = os.path.join(self.swbd_dir, 'terminals', '.'.join([conversation, speaker, 'terminals', 'xml']))
            term_file = open(term_path, "r")
            term_contents = term_file.read()
            term_soup = BeautifulSoup(term_contents, 'lxml')
            terminals = term_soup.find_all('word')
            for terminal in terminals:
                terminal_num = terminal['nite:id']
                start = terminal['nite:start']
                pos = terminal['pos']
                if not start=='non-aligned':
                    term_id = SwbdPreprocessor.get_id(conversation, terminal_num)
                    phonwords = terminal.find_all('nite:pointer')
                    if phonwords:
                        pw_num = SwbdPreprocessor.extract_id_from_href(phonwords[0]['href'])
                        pw_id = SwbdPreprocessor.get_id(conversation,pw_num)
                        pw2term[pw_id] = term_id
                        term2pw[term_id] = pw_id
                        self.tok2pos[pw_id] = pos

            # Put all the phonwords in the appropriate dictionaries. Ignore terminals that are not aligned with phonwords
            pw_path = os.path.join(self.swbd_dir, 'phonwords', '.'.join([conversation, speaker, 'phonwords', 'xml']))
            pw_file = open(pw_path,'r')
            pw_contents = pw_file.read()
            pw_soup = BeautifulSoup(pw_contents, 'lxml')
            pws = pw_soup.find_all('phonword')
            for pw in pws:
                pw_num = pw['nite:id']
                pw_id = SwbdPreprocessor.get_id(conversation,pw_num)

                if pw_id in pw2term:
                    self.tok2tokstr[pw_id] = SwbdPreprocessor.text_reg(pw['orth'])
                    self.tok2times[pw_id] = (float(pw['nite:start']),float(pw['nite:end']))

                    if pw_id in self.correction:
                        self.tok2times[pw_id] = (float(self.correction[pw_id]),float(pw['nite:end']))
                    self.tok2conv[pw_id] = conversation
                    if conversation in self.conv2tok:
                        self.conv2tok[conversation].append(pw_id)
                    else:
                        self.conv2tok[conversation] = [pw_id]

            syntax_path = os.path.join(self.swbd_dir, 'syntax', '.'.join([conversation, speaker, 'syntax', 'xml']))
            syn_file = open(syntax_path, "r")
            syn_contents = syn_file.read()
            syn_soup = BeautifulSoup(syn_contents, 'lxml')
            sentences = syn_soup.find_all('parse')

            # Open the sentence file, and pull out all the sentences and tokens in those sentences
            for sentence in sentences:
                sentence_num = sentence['nite:id']
                utt_id = SwbdPreprocessor.get_id(conversation,sentence_num)
                sentence_terminals = sentence.find_all('nite:child')
                for terminal in sentence_terminals:
                    terminal_num = SwbdPreprocessor.extract_id_from_href(terminal['href'])
                    term_id =  SwbdPreprocessor.get_id(conversation,terminal_num)
                    if term_id in term2pw:
                        pw_id = term2pw[term_id]
                        if utt_id in self.utt2toks:
                            gap = self.tok2times[pw_id][0] - self.tok2times[self.utt2toks[utt_id][-1]][-1]
                            if gap > self.gap_threshold:
                                utt_id = self.utt_id_incr(utt_id)
                        if conversation in self.conv2utt:
                            if not utt_id == self.conv2utt[conversation][-1]:
                                self.conv2utt[conversation].append(utt_id)
                        else:
                            self.conv2utt[conversation] = [utt_id]

                        self.utt2spk[utt_id] = speaker
                        self.utt2conv[utt_id] = conversation

                        if pw_id in self.tok2tokstr: # check that it's a word, not a silence or a contraction
                            self.tok2spk[pw_id] = speaker
                            if utt_id in self.utt2toks:
                                self.utt2toks[utt_id].append(pw_id)
                            else:
                                self.utt2toks[utt_id] = [pw_id]
                            self.tok2utt[pw_id] = utt_id


            # Now find infostatus for tokens

            # First use the syntax file to map non-terminals to terminals
            nt2term = {}
            syn_non_terminals = syn_soup.find_all('nt')
            for nt in syn_non_terminals:
                nt_id = nt['nite:id']
                terms = nt.find_all('nite:child')
                for term in terms:
                    term_num = SwbdPreprocessor.extract_id_from_href(term['href'])
                    term_id = SwbdPreprocessor.get_id(conversation,term_num)
                    if nt_id in nt2term:
                        nt2term[nt_id].append(term_id)
                    else:
                        nt2term[nt_id] = [term_id]

            # Then go through the markables files and assign the infostat to terminals
            infostruc_path = os.path.join(self.swbd_dir, 'markable', '.'.join([conversation, speaker, 'markable', 'xml']))
            infostruc_file = open(infostruc_path,'r')
            infostruc_contents = infostruc_file.read()
            infostruc_soup = BeautifulSoup(infostruc_contents,'lxml')
            markables = infostruc_soup.find_all('markable')
            for markable in markables:
                try:
                    infostat = markable['status']
                    if not infostat in self.allowed_infostats:
                        infostat = None
                except:
                    infostat = None
                syn_nt = SwbdPreprocessor.extract_id_from_href(markable.find_all('nite:pointer')[0]['href'])
                if syn_nt in nt2term:
                    for term in nt2term[syn_nt]:
                        if term in term2pw:
                            if term2pw[term] in self.tok2utt: self.tok2infostat[term2pw[term]] = infostat
                elif syn_nt in term2pw: # sometimes the markable is marked on a terminal, not a non-terminal
                    if term2pw[term] in self.tok2utt: self.tok2infostat[term2pw[syn_nt]] = infostat  # added condition that tok has to be in tok2utt
            for tok in self.conv2tok[conversation]:
                if tok not in self.tok2infostat and tok in self.tok2utt: # added condition that tok has to be in tok2utt
                    self.tok2infostat[tok] = None

            accent_path = os.path.join(self.swbd_dir, 'accent', '.'.join([conversation, speaker, 'accents', 'xml']))
            if os.path.exists(accent_path):
                accent_file = open(accent_path,'r')
                accent_contents = accent_file.read()
                accent_soup = BeautifulSoup(accent_contents,'lxml')
                accents = accent_soup.find_all('accent')
                for accent in accents:
                    pw_num = SwbdPreprocessor.extract_id_from_href(accent.find_all('nite:pointer')[0]['href'])
                    pw_id = SwbdPreprocessor.get_id(conversation,pw_num)
                    if pw_id in self.tok2utt: self.tok2tone[pw_id] = 1  # added condition that tok has to be in tok2utt
                for pw in self.conv2tok[conversation]:
                    if pw not in self.tok2tone:
                        if pw_id in self.tok2utt: self.tok2tone[pw] = 0  # added condition that tok has to be in tok2utt

            kontrast_path = os.path.join(self.swbd_dir, 'kontrast', '.'.join([conversation, 'kontrast', 'xml']))
            found = 0
            not_found = 0
            if os.path.exists(kontrast_path):
                kontrast_file = open(kontrast_path, 'r')
                kontrast_contents = kontrast_file.read()
                kontrast_soup = BeautifulSoup(kontrast_contents, 'lxml')
                kontrasts = kontrast_soup.find_all('kontrast')
                for kontrast in kontrasts:
                    kontrast_type = kontrast['type']
                    terms = kontrast.find_all('nite:child')
                    term_ids = ['_'.join([conversation,SwbdPreprocessor.extract_id_from_href(term['href'])]) for term in terms]
                    for term_id in term_ids:
                        if term_id in term2pw:
                            self.tok2kontrast[term2pw[term_id]] = kontrast_type

                for tok in self.conv2tok[conversation]:
                    if conversation == 'sw2295':
                        import pdb;pdb.set_trace()
                    if tok not in self.tok2kontrast and tok in self.tok2utt:  # added condition that tok has to be in tok2utt
                        self.tok2kontrast[tok] = None
        print('a')

        self.utt_ids = list(self.utt2toks.keys())
        broken_toks = []
        for tok in self.tok2times:
            if self.tok2times[tok][0]==self.tok2times[tok][1]:
                broken_toks.append(tok)

        print('b')
        for utt_id in self.utt2toks:
            utt_start = self.tok2times[self.utt2toks[utt_id][0]][0]
            self.utt2tokentimes[utt_id] = [float(self.tok2times[tok][0]) for tok in self.utt2toks[utt_id]] + [self.tok2times[self.utt2toks[utt_id][-1]][-1]]
            self.utt2startend[utt_id] = (self.utt2tokentimes[utt_id][0],self.utt2tokentimes[utt_id][-1])
            self.utt2text[utt_id] = [self.tok2tokstr[tok] for tok in self.utt2toks[utt_id]]
            self.utt2frames[utt_id] = torch.tensor([int(round(float(tim-utt_start)*100)) for tim in self.utt2tokentimes[utt_id]],dtype=torch.float32)
        print('c')
        self.make_BIO()
        print('d')
        self.make_new_tags()
        print('e')
        self.make_old_tags()
        print('f')
        self.make_kontrast_tags()
        print('g')
        
    def make_new_tags(self):
        for utt in self.utt2toks:
            toks = [tok for tok in self.utt2toks[utt]]
            tags = [self.tok2infostat[tok] for tok in toks]
            newness = [1 if tag=='new' else 0 for tag in tags]
            self.utt2new[utt] = newness

    def make_old_tags(self):
        for utt in self.utt2toks:
            toks = [tok for tok in self.utt2toks[utt]]
            tags = [self.tok2infostat[tok] for tok in toks]
            oldness = [1 if tag=='old' else 0 for tag in tags]
            self.utt2old[utt] = oldness

    def make_BIO(self):
        for utt in self.utt2toks:
            b = 'B-'
            i = 'I-'
            o = 'O'
            toks = [tok for tok in self.utt2toks[utt]]
            tags = [self.tok2infostat[tok] for tok in toks]
            if tags[0]==None:
                bio_tags = [o]
            else:
                bio_tags = [b+tags[0]]
            for j in range(1,len(tags)):
                if tags[j] == None:
                    bio_tags.append(o)
                elif tags[j] == tags[j-1]:
                    bio_tags.append(i+tags[j])
                else:
                    bio_tags.append(b+tags[j])

            self.utt2bio[utt] = bio_tags

    def make_kontrast_tags(self):
        for utt in self.utt2toks:
            conv = self.utt2conv[utt]
            kontrast_path = f'{self.swbd_dir}/kontrast/{conv}.kontrast.xml'
            if os.path.exists(kontrast_path):
                kontrasts = [self.tok2kontrast[tok] for tok in self.utt2toks[utt]]
                kontrast_tags = [0 if kontrast=='background' or kontrast==None else 1 for kontrast in kontrasts]
                self.utt2kontrast[utt] = kontrast_tags

    def get_tok_feats(self,df,start,end):
        tok_df = df.loc[df['frameTime'] >= start]
        tok_df = tok_df.loc[tok_df['frameTime'] < end]
        #import pdb;pdb.set_trace()
        tok_df = tok_df[self.pros_feat_names]
        if len(tok_df)==0:
            return torch.tensor([])
        feat_tensors = []
        for i, row in tok_df.iterrows():
            tens = torch.tensor(row.tolist()).view(1, len(row.tolist()))
            feat_tensors.append(tens)
        try:
            return torch.cat(feat_tensors,dim=0)
        except:
            import pdb;pdb.set_trace()

    def load_opensmile_feats(self):
        for conv in self.conv2utt:
            filename = '.'.join([conv,'is13','csv'])
            feat_df = pd.read_csv(os.path.join(self.pros_feat_dir, filename), sep=';')
            feat_df = feat_df[['frameTime'] + self.pros_feat_names]
            for utt in self.conv2utt[conv]:
                for tok in self.utt2toks[utt]:
                    print(tok)
                    tok_start = self.tok2times[tok][0]
                    tok_end = self.tok2times[tok][1]
                    self.tok2prosfeats[tok] = self.get_tok_feats(feat_df, tok_start, tok_end)

    def acoustic_preproc(self):
        self.load_opensmile_feats()



    def preproc(self,write_dict=True,out_file='swbd.pkl',acc_only=False,kontrast_only=False):
        with open(self.annotated_files,'r') as f:
            file_list = f.readlines()
        self.text_preproc(file_list)
        self.acoustic_preproc()
        self.gen_nested_dict(acc_only,kontrast_only)
        if write_dict:
            self.save_nested(name=out_file)


def main():

    swbd_dir = '/afs/inf.ed.ac.uk/user/s18/s1899827/xml'
    pros_feat_dir = '~/opensmile-2.3.0/swbd2'
    annotated_files = '../data/swbd/annotated_files.txt'
    save_dir = '../data/swbd_kontrast'
    preprocessor = SwbdPreprocessor(swbd_dir,pros_feat_dir,save_dir,annotated_files)

    # acc_only: only save instances that are annotated with accent info
    # kontrast_only: only save instances that are annotated with kontrast info
    # default is to save all instances that are annotated with new/mediated/old.
    # This is a superset of the other two.
    preprocessor.preproc(acc_only=False,kontrast_only=False, out_file='swbd.pkl')

if __name__=="__main__":
    main()

