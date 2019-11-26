"""
Prepare BU radio news corpus data for kaldi feat extraction

Modeled on Herman Kamper's recipe for Mboshi parallel data
"""

import re
import glob
import os
import subprocess
from string import punctuation
import nltk
nltk.download('punkt')

#TOKENIZATION_METHOD = 'default'
#TOKENIZATION_METHOD = 'breath_sent'
TOKENIZATION_METHOD = 'breath_tok'

class BurncPreprocessor:
    def __init__(self,burnc_dir,out_dir,speakers_file):

        self.burnc_dir = burnc_dir
        self.out_dir = out_dir
        self.speakers_file = speakers_file

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

        # adding the following 26 Nov
        self.para2utt = {}
        self.utt2toks = {}
        self.tok2tones = {}
        self.tok2times = {}
        self.tok2tokstr = {}

    @staticmethod
    def text_reg(word):
        remove = punctuation.replace('-','').replace('<','').replace('>','')
        word = word.lower().replace("'s","").replace("n't","").replace('/n','').replace('/v','')
        word = word.translate(str.maketrans('', '', remove))
        return word

    @staticmethod
    def load_word_file(wdfile):
        with open(wdfile, 'r',encoding="utf8", errors='ignore') as f:
            annotated_wds = f.read().split('#')[1]
            lines = [line.strip() for line in annotated_wds.split('\n') if not line == '']
            if lines:
                if len(lines[0].split())<3:
                    lines = lines[1:]
                words = [BurncPreprocessor.text_reg(line.split()[2]) for line in lines]
                timestamps = [float(line.split()[0]) for line in lines]
                timestamps = [0] + timestamps
                return words,lines,timestamps
            else:
                return None,None,None

    @staticmethod
    def load_text_file(txtfile):
        break_pairs = []
        with open(txtfile, 'r',encoding="utf8", errors='ignore') as f:
            text = f.readlines()
            text = ' '.join(text).replace('\n', '').lower()
            if TOKENIZATION_METHOD=='default':
                re_break = r'([a-zA-z]+)[\.\?!][\s]+brth[\s]+([a-zA-z]+)'  # default
            elif TOKENIZATION_METHOD=='breath_tok':
                re_break = r'([a-zA-z]+)[\s]*[\.\?!]?[\s]+brth[\s]+([a-zA-z]+)' # breath_tok
            m = re.findall(re_break, text)
            break_pairs.extend(m)
        return break_pairs

    @staticmethod
    def flatten_list(in_list):
        flat_list = []
        for sublist in in_list:
            for item in sublist:
                list_item = BurncPreprocessor.text_reg(item).strip()
                if list_item:
                    flat_list.append(list_item)
        return flat_list

    @staticmethod
    def load_text_file_nltk(txtfile): # breath_sent
        from nltk.tokenize import sent_tokenize
        break_pairs = []
        with open(txtfile, 'r',encoding="utf8", errors='ignore') as f:
            text = f.readlines()
            text = ' '.join(text).replace('\n', '').replace('-',' ').lower()
            sents = sent_tokenize(text)
            sents = BurncPreprocessor.flatten_list([sent.strip().split('brth') for sent in sents])
            for i in range(len(sents)-1):
                break_pairs.append((sents[i].split()[-1],sents[i+1].split()[0]))
        return break_pairs

    @staticmethod
    def collapse_double_tones(times,tones):
        '''
        Some tones are double annotated, in which case they have the same timestamp.
        Go through the list of times, looking for duplicate times.
        Delete duplicate time that does not have an asterisk in the tone
        (either one if neither one has an asterisk)
        '''
        dup_idx = []
        try:
            assert(len(times)==len(tones))
        except:
            print("Can't collapse tones given time and tone lists of diff lengths!")
        # Go through times and keep track of the index duplicated times
        for i in range(len(times)-1):
            if times[i]==times[i+1]:
                dup_idx.append((i,i+1))
        # Go through the indices of the duplicated times and pick
        # out which one is deletable (doesn't contain a *)
        del_idx = []
        for pair in dup_idx:
            if not '*' in tones[pair[0]]:
                del_idx.append(pair[0])
            else:
                del_idx.append(pair[1])
        out_times = []
        out_tones = []
        for i in range(len(times)):
            if not i in del_idx:
                out_times.append(times[i])
                out_tones.append(tones[i])
        return out_times,out_tones

    @staticmethod
    def load_tone_file(tonefile):
        time2tone = None
        with open(tonefile, 'r',encoding="utf8", errors='ignore') as f:
            tone_annot = f.read().split('#')[1]
            lines = [line.strip() for line in tone_annot.split('\n') if not line == '']
            if lines:
                if len(lines[0].split()) < 3:
                    lines = lines[1:]
                times = [float(line.split()[0]) for line in lines]
                tones = [line.split()[2] for line in lines]
                times,tones = BurncPreprocessor.collapse_double_tones(times,tones)
                bin_tones = [1 if '*' in tone else 0 for tone in tones]
                time2tone = dict(zip(times,bin_tones))
            return time2tone

    @staticmethod
    def convert_to_sec(timestamp):
        h, m, s = timestamp.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)

    def write_segments(self):
        print('write segments')
        with open(os.path.join(self.out_dir,'segments'),'w') as f:
            for utt in self.utterances:
                f.write(utt+" "+self.utt2recording[utt]+" "+str(self.utt2startend[utt][0])+" "+str(self.utt2startend[utt][1]))
                f.write('\n')

    def sort_rec(self):
        recordingids = sorted(list(self.recording2file.keys()))
        return recordingids

    def write_wav_scp(self):
        recordingids = self.sort_rec(self.recording2file)
        with open(os.path.join(self.out_dir,'wav.scp'),'w') as f:
            for recording in recordingids:
                f.write(recording+" /home/elizabeth/repos/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 "+self.recording2file[recording]+" |")
                f.write('\n')

    def write_text(self):
        print('write utt2text')
        with open(os.path.join(self.out_dir,'text'),'w') as f:
            for utt in self.utterances:
                f.write(utt+" "+self.utt2text[utt])
                f.write('\n')

    def write_utt2spk(self):
        print('write utt2spk')
        with open(os.path.join(self.out_dir,'utt2spk'),'w') as f:
            for utt in self.utterances:
                f.write(utt+" "+self.utt2spk[utt])
                f.write('\n')

    @staticmethod
    def map_tones_to_words(time2tone,timestamps,words):
        tones_per_word = [0 for wd in words]
        for i in range(len(timestamps) - 1):
            for time in time2tone:
                if time < timestamps[i+1] and time >= timestamps[i]:
                    if tones_per_word[i]==0:
                        tones_per_word[i] = time2tone[time]
        return tones_per_word

    def write_text2labels(self):
        print('write text2labels')
        with open(os.path.join(self.out_dir,'text2labels'),'w') as f:
            for utt in self.utterances:
                f.write(utt+"\t"+self.utt2text[utt]+"\t"+' '.join([str(tone) for tone in self.utt2tones[utt]]))
                f.write('\n')

    def write_utt2toktime(self):
        with open(os.path.join(self.out_dir,'utt2toktimes'),'w') as f:
            for utt in self.utterances:
                f.write(utt+"\t"+' '.join([str(tim) for tim in self.utt2toktimes[utt]]))
                f.write("\n")

    """
    @staticmethod
    def make_three_tok_spans(para_id,words,tones,timestamps):
        spans = []
        padded_words = ['<PAD>'] + words + ['<PAD>']
        padded_tones = [0] + tones + [0]
        timestamps = [timestamps[0]] + timestamps + [timestamps[-1]]
        for i in range(1,len(padded_words)-1):
            toktimes = (timestamps[i-1],timestamps[i],timestamps[i+1],timestamps[i+2])
            spans.append((para_id,(padded_words[i-1],padded_words[i],padded_words[i+1]),padded_tones[i],toktimes))
        return spans

    def write_three_tok_spans(self):
        with open(os.path.join(self.out_dir, 'spans'), 'w') as f:
            for para_id,span,label,toktimes in self.spans:
                f.write(para_id+'\t'+' '.join(span)+'\t'+str(label)+'\t'+' '.join([str(tim) for tim in toktimes]))
                f.write('\n')
    """

    def write_kaldi_inputs(self):
        self.write_segments()
        self.write_wav_scp()
        self.write_text()
        self.write_utt2spk()
        self.write_text2labels()
        self.write_utt2toktime()
        self.write_three_tok_spans()

    def process_utt(self,words,timestamps,para_id,spkr,recordingid,tones_per_word,idx):
        # Make an utterance id: paragraph id + start time + end time
        utt_start = timestamps[0]
        utt_end = timestamps[idx + 1]
        utt_id = para_id + '-' + '%08.3f' % utt_start + '-' + '%08.3f' % utt_end

        self.utt_ids.append(utt_id)

        # Make dict of para_id: [utt1, utt2, ...]
        if not para_id in self.para2utt:
            self.para2utt[para_id] = [utt_id]
        else:
            self.para2utt[para_id].append(utt_id)

        curr_toks = words[:idx + 1]
        curr_toktimes = timestamps[:idx + 2]
        curr_tones = tones_per_word[:idx + 1]
        for i, tok in enumerate(curr_toks):
            tok_start = curr_toktimes[i]
            tok_end = curr_toktimes[i + 1]
            tok_id = para_id + '-tok-' + '%08.3f' % tok_start + '-' + '%08.3f' % tok_end
            self.tok2tokstr[tok_id] = tok
            self.tok2times[tok_id] = (tok_start, tok_end)
            self.tok2tones[tok_id] = curr_tones[i]
            if not utt_id in self.utt2toks:
                self.utt2toks[utt_id] = [tok_id]
            else:
                self.utt2toks[utt_id].append(tok_id)

        self.utt2text[utt_id] = ' '.join(curr_toks)
        self.utt2spk[utt_id] = spkr
        self.utt2recording[utt_id] = recordingid
        self.utt2startend[utt_id] = (utt_start, utt_end)  # store start and end timestamp of utterance
        self.utt2tokentimes[utt_id] = curr_toktimes
        self.utt2tones[utt_id] = curr_tones

    def text_preproc(self):
        # Segment text into sentence-level utterances

        with open(self.speakers_file,'r') as f:
            speakers = [line.strip() for line in f.readlines()]
        # Go through all the datafiles
        for sp in speakers:
            datadir = os.path.join(self.burnc_dir,sp)
            for subdir, dirs, files in os.walk(datadir):
                for file in files:

                    # For each distinct paragraph, pull out the word file, text file, and recording file
                    if '.wrd' in file:
                        para_id = file.split('.')[0]
                        self.para_ids.append(para_id)
                        print(para_id)
                        wordfile = os.path.join(subdir,file)
                        textfile = os.path.join(subdir,para_id+'.txt')
                        if not os.path.exists(textfile):
                            textfile = os.path.join(subdir,para_id+'.txn')

                        # Load tone file
                        tonefile = os.path.join(subdir,para_id+'.ton')

                        if os.path.exists(tonefile):

                            # Open the tone file and load dictionary of time to tone value (0 or 1)
                            time2tone = BurncPreprocessor.load_tone_file(tonefile)

                            # Open the word file and load in as two lists -- one of words, one of timestamps of beginnings of words
                            words,lines,timestamps = BurncPreprocessor.load_word_file(wordfile)

                            if words and time2tone:


                                # Load recording file
                                recordingid = para_id
                                recordingfile = os.path.join(subdir, para_id + '.sph')

                                if not os.path.exists(recordingfile):
                                    recordingfile = os.path.join(subdir, para_id + '.spn')

                                self.recording2file[recordingid] = recordingfile

                                # Convert tone dict to a list of same len as words, with all words either 0 or 1
                                tones_per_word = BurncPreprocessor.map_tones_to_words(time2tone,timestamps,words)

                                if sum(tones_per_word)==0:
                                    print('NO TONES')
                                    import pdb;pdb.set_trace()

                                # While you're here, make 3-token spans for replicating Stehwien et al.:
                                # three_tok_spans.extend(make_three_tok_spans(para_id,words,tones_per_word,timestamps))

                                # Open the text file and break on sentence breaks followed by breaths.
                                # Store breaks as pairs of words -- one on either side of the break.
                                # This requires less match-up between the words file and the text file,
                                # which are inconsistent with one another.
                                if TOKENIZATION_METHOD=='breath_sent':
                                    break_pairs = BurncPreprocessor.load_text_file_nltk(textfile) #breath_sent
                                elif TOKENIZATION_METHOD=='default' or TOKENIZATION_METHOD=='breath_tok':
                                    break_pairs = BurncPreprocessor.load_text_file(textfile) # default or breath_tok
                                else:
                                    print('Tokenization method not given or not recognized')
                                    import pdb;pdb.set_trace()

                                # Now use the break pairs to segment the text
                                utt_list = []
                                utt_token_times = []
                                utt_start_end = []
                                utt_labels = []
                                for break_pair in break_pairs:
                                    idx = 0

                                    while idx < len(words)-1:
                                        if words[idx].strip() == break_pair[0].strip() and \
                                            words[idx+1].strip() == break_pair[1].strip():

                                            self.process_utt(words, timestamps, para_id, sp, recordingid, tones_per_word, idx)
                                            """
    
                                            # Make an utterance id: paragraph id + start time + end time
                                            utt_start = timestamps[0]
                                            utt_end = timestamps[idx + 1]
                                            utt_id = para_id+'-'+'%08.3f'%utt_start+'-'+ '%08.3f'%utt_end
    
                                            utt_ids.append(utt_id)
    
                                            # Make dict of utt: (start time, end time)
                                            utt_start_end.append((utt_start,utt_end))
    
                                            # Make dict of para_id: [utt1, utt2, ...]
                                            if not para_id in para2utt:
                                                para2utt[para_id] = [utt_id]
                                            else:
                                                para2utt[para_id].append(utt_id)
    
                                            curr_toks = words[:idx+1]
                                            curr_toktimes = timestamps[:idx+2]
                                            curr_tones = tones_per_word[:idx+1]
                                            for i,tok in enumerate(curr_toks):
                                                tok_start = curr_toktimes[i]
                                                tok_end = curr_toktimes[i+1]
                                                tok_id = para_id+'-tok-'+'%08.3f'%tok_start+'-'+ '%08.3f'%tok_end
                                                tok2tokstr[tok_id] = tok
                                                tok2times[tok_id] = (tok_start,tok_end)
                                                tok2tones[tok_id] = curr_tones[i]
                                                if not utt_id in utt2toks:
                                                    utt2toks[utt_id] = [tok_id]
                                                else:
                                                    utt2toks[utt_id].append(tok_id)
    
    
                                            utt_list.append(curr_toks)
                                            utt_token_times.append(curr_toktimes)
                                            utt_labels.append(curr_tones)
                                            """

                                            # Chop the consumed words/times off the front of those lists
                                            words = words[idx+1:]
                                            timestamps = timestamps[idx+1:]
                                            tones_per_word = tones_per_word[idx+1:]
                                            break

                                        else:
                                            idx += 1

                                self.process_utt(words, timestamps, para_id, sp, recordingid, tones_per_word,  idx=len(words)-1)

                                """
                                # Last utterance:
                                # TODO add word stuff here or somehow take this bit out
                                curr_toks = words
                                curr_toktimes = timestamps
                                curr_tones = tones_per_word
    
                                utt_list.append(curr_toks)
                                utt_token_times.append(curr_toktimes)
                                utt_start = timestamps[0]
                                utt_end = timestamps[-1]
                                utt_id = para_id + '-' + '%08.3f' % utt_start + '-' + '%08.3f' % utt_end
                                utt_ids.append(utt_id)
                                utt_start_end.append((utt_start, utt_end))
                                utt_labels.append(tones_per_word)
    
                                for i, tok in enumerate(curr_toks):
                                    tok_start = curr_toktimes[i]
                                    tok_end = curr_toktimes[i + 1]
                                    tok_id = para_id + '-tok-' + '%08.3f' % tok_start + '-' + '%08.3f' % tok_end
                                    tok2tokstr[tok_id] = tok
                                    tok2times[tok_id] = (tok_start, tok_end)
                                    tok2tones[tok_id] = curr_tones[i]
                                    if not utt_id in utt2toks:
                                        utt2toks[utt_id] = [tok]
                                    else:
                                        utt2toks[utt_id].append(tok)
                                """







def main():
    speakers_file = 'burnc_speakers.txt'
    burnc_dir = "/home/elizabeth/repos/kaldi/egs/burnc/kaldi_features/data"
    out_dir = 'tmp'

    preprocessor = BurncPreprocessor(burnc_dir,out_dir,speakers_file)
    preprocessor.text_preproc()
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    main()
