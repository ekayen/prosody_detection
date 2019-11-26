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

def text_reg(word):
    remove = punctuation.replace('-','').replace('<','').replace('>','')
    word = word.lower().replace("'s","").replace("n't","").replace('/n','').replace('/v','')
    word = word.translate(str.maketrans('', '', remove))
    return word

def load_word_file(wdfile):
    with open(wdfile, 'r',encoding="utf8", errors='ignore') as f:
        annotated_wds = f.read().split('#')[1]
        lines = [line.strip() for line in annotated_wds.split('\n') if not line == '']
        if lines:
            if len(lines[0].split())<3:
                lines = lines[1:]
            words = [text_reg(line.split()[2]) for line in lines]
            timestamps = [float(line.split()[0]) for line in lines]
            timestamps = [0] + timestamps
            return words,lines,timestamps
        else:
            return None,None,None

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

def flatten_list(in_list):
    flat_list = []
    for sublist in in_list:
        for item in sublist:
            list_item = text_reg(item).strip()
            if list_item:
                flat_list.append(list_item)
    return flat_list

def load_text_file_nltk(txtfile): # breath_sent
    from nltk.tokenize import sent_tokenize
    break_pairs = []
    with open(txtfile, 'r',encoding="utf8", errors='ignore') as f:
        text = f.readlines()
        text = ' '.join(text).replace('\n', '').replace('-',' ').lower()
        sents = sent_tokenize(text)
        sents = flatten_list([sent.strip().split('brth') for sent in sents])
        for i in range(len(sents)-1):
            break_pairs.append((sents[i].split()[-1],sents[i+1].split()[0]))
    return break_pairs

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
            times,tones = collapse_double_tones(times,tones)
            bin_tones = [1 if '*' in tone else 0 for tone in tones]
            time2tone = dict(zip(times,bin_tones))
        return time2tone

def convert_to_sec(timestamp):
    h, m, s = timestamp.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def write_segments(utt2startend,utt2recording,utterances,out_dir):
    print('write segments')
    with open(os.path.join(out_dir,'segments'),'w') as f:
        for utt in utterances:
            f.write(utt+" "+utt2recording[utt]+" "+str(utt2startend[utt][0])+" "+str(utt2startend[utt][1]))
            f.write('\n')

def sort_rec(recording2file):
    recordingids = sorted(list(recording2file.keys()))
    return recordingids

def write_wav_scp(recording2file,utterances,out_dir):
    recordingids = sort_rec(recording2file)
    with open(os.path.join(out_dir,'wav.scp'),'w') as f:
        for recording in recordingids:
            f.write(recording+" /home/elizabeth/repos/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 "+recording2file[recording]+" |")
            f.write('\n')


def write_text(utt2text,utterances,out_dir):
    print('write utt2text')
    with open(os.path.join(out_dir,'text'),'w') as f:
        for utt in utterances:
            f.write(utt+" "+utt2text[utt])
            f.write('\n')

def write_utt2spk(utt2spk,utterances,out_dir):
    print('write utt2spk')
    with open(os.path.join(out_dir,'utt2spk'),'w') as f:
        for utt in utterances:
            f.write(utt+" "+utt2spk[utt])
            f.write('\n')

def map_tones_to_words(time2tone,timestamps,words):
    tones_per_word = [0 for wd in words]
    for i in range(len(timestamps) - 1):
        for time in time2tone:
            if time < timestamps[i+1] and time >= timestamps[i]:
                if tones_per_word[i]==0:
                    tones_per_word[i] = time2tone[time]
    return tones_per_word

def write_text2labels(utt2text,utt2tones,utterances,out_dir):
    print('write text2labels')
    with open(os.path.join(out_dir,'text2labels'),'w') as f:
        for utt in utterances:
            f.write(utt+"\t"+utt2text[utt]+"\t"+' '.join([str(tone) for tone in utt2tones[utt]]))
            f.write('\n')

def write_utt2toktime(utt2toktimes,utterances,out_dir):
    with open(os.path.join(out_dir,'utt2toktimes'),'w') as f:
        for utt in utterances:
            f.write(utt+"\t"+' '.join([str(tim) for tim in utt2toktimes[utt]]))
            f.write("\n")

def make_three_tok_spans(para_id,words,tones,timestamps):
    spans = []
    padded_words = ['<PAD>'] + words + ['<PAD>']
    padded_tones = [0] + tones + [0]
    timestamps = [timestamps[0]] + timestamps + [timestamps[-1]]
    for i in range(1,len(padded_words)-1):
        toktimes = (timestamps[i-1],timestamps[i],timestamps[i+1],timestamps[i+2])
        spans.append((para_id,(padded_words[i-1],padded_words[i],padded_words[i+1]),padded_tones[i],toktimes))
    return spans

def write_three_tok_spans(spans,out_dir):
    with open(os.path.join(out_dir, 'spans'), 'w') as f:
        for para_id,span,label,toktimes in spans:
            f.write(para_id+'\t'+' '.join(span)+'\t'+str(label)+'\t'+' '.join([str(tim) for tim in toktimes]))
            f.write('\n')

def gen_kaldi_inputs(burnc_dir,out_dir,speakers_file):
    # Segment text into sentence-level utterances
    three_tok_spans = []
    speakers_used = set()
    utterances = []
    utt2text = {}
    utt2spk = {}
    utt2recording = {}
    recording2file = {}
    utt2startend = {} # store start and end timestamp of utterance
    utt2tokentimes = {} # store start time of each token (not necessary for kaldi, but plan to use in model)
    utt2tones = {}
    with open(speakers_file,'r') as f:
        speakers = [line.strip() for line in f.readlines()]
    # Go through all the datafiles
    for sp in speakers:
        datadir = os.path.join(burnc_dir,sp)
        for subdir, dirs, files in os.walk(datadir):
            for file in files:

                # For each distinct paragraph, pull out the word file, text file, and recording file
                if '.wrd' in file:
                    para_id = file.split('.')[0]
                    print(para_id)
                    wordfile = os.path.join(subdir,file)
                    textfile = os.path.join(subdir,para_id+'.txt')
                    if not os.path.exists(textfile):
                        textfile = os.path.join(subdir,para_id+'.txn')

                    # Load tone file
                    tonefile = os.path.join(subdir,para_id+'.ton')

                    if os.path.exists(tonefile):

                        # Open the tone file and load dictionary of time to tone value (0 or 1)
                        time2tone = load_tone_file(tonefile)

                        # Open the word file and load in as two lists -- one of words, one of timestamps of beginnings of words
                        words,lines,timestamps = load_word_file(wordfile)

                        if words and time2tone:

                            speakers_used.add(sp)

                            # Load recording file
                            recordingid = para_id
                            recordingfile = os.path.join(subdir, para_id + '.sph')

                            if not os.path.exists(recordingfile):
                                recordingfile = os.path.join(subdir, para_id + '.spn')

                            recording2file[recordingid] = recordingfile

                            # Convert tone dict to a list of same len as words, with all words either 0 or 1
                            tones_per_word = map_tones_to_words(time2tone,timestamps,words)

                            if sum(tones_per_word)==0:
                                print('NO TONES')
                                import pdb;pdb.set_trace()

                            # While you're here, make 3-token spans for replicating Stehwien et al.:
                            three_tok_spans.extend(make_three_tok_spans(para_id,words,tones_per_word,timestamps))

                            # Open the text file and break on sentence breaks followed by breaths.
                            # Store breaks as pairs of words -- one on either side of the break.
                            # This requires less match-up between the words file and the text file,
                            # which are inconsistent with one another.
                            if TOKENIZATION_METHOD=='breath_sent':
                                break_pairs = load_text_file_nltk(textfile) #breath_sent
                            elif TOKENIZATION_METHOD=='default' or TOKENIZATION_METHOD=='breath_tok':
                                break_pairs = load_text_file(textfile) # default or breath_tok
                            else:
                                print('Tokenization method not given or not recognized')
                                import pdb;pdb.set_trace()

                            # Now use the break pairs to segment the text
                            utt_list = []
                            utt_token_times = []
                            utt_ids = []
                            utt_start_end = []
                            utt_labels = []
                            for break_pair in break_pairs:
                                idx = 0

                                while idx < len(words)-1:
                                    if words[idx].strip() == break_pair[0].strip() and \
                                        words[idx+1].strip() == break_pair[1].strip():

                                        utt_list.append(words[:idx+1])
                                        utt_token_times.append(timestamps[:idx+2])
                                        utt_labels.append(tones_per_word[:idx+1])

                                        # Make an utterance id: paragraph id + start time + end time
                                        start_time = timestamps[0]
                                        end_time = timestamps[idx+1]
                                        utt_ids.append(para_id+'-'+'%08.3f'%start_time+'-'+ '%08.3f'%end_time)
                                        utt_start_end.append((start_time,end_time))

                                        # Chop the consumed words/times off the front of those lists
                                        words = words[idx+1:]
                                        timestamps = timestamps[idx+1:]
                                        tones_per_word = tones_per_word[idx+1:]
                                        break

                                    else:
                                        idx += 1

                            utt_list.append(words)
                            utt_token_times.append(timestamps)
                            start_time = timestamps[0]
                            end_time = timestamps[-1]
                            utt_ids.append(para_id + '-' + '%08.3f' % start_time + '-' + '%08.3f' % end_time)
                            utt_start_end.append((start_time, end_time))
                            utt_labels.append(tones_per_word)

                            for i,utt_id in enumerate(utt_ids):
                                utt2text[utt_id] = ' '.join(utt_list[i])
                                utt2spk[utt_id] = sp
                                utt2recording[utt_id] = recordingid
                                utt2startend[utt_id] = utt_start_end[i]  # store start and end timestamp of utterance
                                utt2tokentimes[utt_id] = utt_token_times[i]
                                utt2tones[utt_id] = utt_labels[i]

    utterances = sorted(list(utt2text.keys()))
    write_segments(utt2startend,utt2recording,utterances,out_dir=out_dir)
    write_wav_scp(recording2file,utterances,out_dir=out_dir)
    write_text(utt2text,utterances,out_dir=out_dir)
    write_utt2spk(utt2spk,utterances,out_dir=out_dir)
    write_text2labels(utt2text,utt2tones,utterances,out_dir=out_dir)
    write_utt2toktime(utt2tokentimes,utterances,out_dir=out_dir)
    write_three_tok_spans(three_tok_spans,out_dir=out_dir)



def main():
    speakers_file = 'burnc_speakers.txt'
    burnc_dir = "/home/elizabeth/repos/kaldi/egs/burnc/kaldi_features/data"
    out_dir = 'tmp'
    gen_kaldi_inputs(burnc_dir,out_dir,speakers_file)


if __name__ == "__main__":
    main()
