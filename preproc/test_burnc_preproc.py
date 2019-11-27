from burnc_preproc import BurncPreprocessor
import pickle
import os

test_from_file = True
save_dir = 'tmp'
data_name = 'proc.pkl'

if test_from_file:
    with open(os.path.join(save_dir,data_name),'rb') as f:
        proc = pickle.load(f)
        nested = proc.nested
else:
    speakers_file = 'burnc_speakers.txt'
    burnc_dir = "/home/elizabeth/repos/kaldi/egs/burnc/kaldi_features/data"
    pros_feat_dir = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/burnc'
    mfcc_dir = '/home/elizabeth/repos/kaldi/egs/burnc/kaldi_features/data/train_breath_tok/feats.scp'
    kaldi_dir = 'tmp'

    proc = BurncPreprocessor(burnc_dir, pros_feat_dir, mfcc_dir, kaldi_dir, speakers_file, save_dir)
    proc.preproc(write_dict=False)
    nested = proc.nested


def test_frame_sizes():
    for para in nested:
        for tok in nested[para]['tok2times']:
            utt = proc.tok2utt[tok]
            if proc.utt2toks[utt][-1]==tok:
                last = True
            else:
                last = False
            span = abs(nested[para]['tok2times'][tok][1]-nested[para]['tok2times'][tok][0])*100
            print(tok,nested[para]['tok2times'][tok],span,nested[para]['mfccs'][tok].shape[0],nested[para]['prosfeats'][tok].shape[0])
            if last:
                assert abs(nested[para]['mfccs'][tok].shape[0] - nested[para]['prosfeats'][tok].shape[0]) <= 4
            else:
                assert abs(nested[para]['mfccs'][tok].shape[0] - nested[para]['prosfeats'][tok].shape[0]) <= 1
            num_mfcc_frames = nested[para]['mfccs'][tok].shape[0]
            num_pros_frames = nested[para]['prosfeats'][tok].shape[0]
            assert abs(num_mfcc_frames - span) <= 5
            assert abs(num_pros_frames - span) <= 5


def test_num_toks():
    assert len(proc.tok2times) == len(proc.tok2tokstr)
    assert len(proc.tok2tone) == len(proc.tok2tokstr)
    assert len(proc.tok2mfccfeats) == len(proc.tok2tokstr)
    assert len(proc.tok2prosfeats) == len(proc.tok2tokstr)
    assert len(proc.tok2utt) == len(proc.tok2tokstr)
    toks_in_utts = 0
    for utt in proc.utt2toks:
        toks_in_utts += len(proc.utt2toks[utt])
    assert len(proc.tok2times)==toks_in_utts

def test_num_utts():
    assert len(proc.utt2text) == len(proc.utt2spk)
    assert len(proc.utt2recording) == len(proc.utt2spk)
    assert len(proc.utt2startend) == len(proc.utt2spk)
    assert len(proc.utt2tokentimes) == len(proc.utt2spk)
    assert len(proc.utt2tones) == len(proc.utt2spk)
    assert len(proc.utt_ids) == len(proc.utt2spk)
    assert len(proc.utt2toks) == len(proc.utt2spk)
    assert len(proc.utt2para) == len(proc.utt2spk)
    utts_in_para = 0
    for para in proc.para2utt:
        utts_in_para += len(proc.para2utt[para])
    assert utts_in_para == len(proc.utt2spk)

test_frame_sizes()
test_num_toks()
test_num_utts()