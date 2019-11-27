from burnc_preproc import BurncPreprocessor
from paths import burnc_dir,kaldi_dir,speakers_file


def main():
    speakers_file = 'burnc_speakers.txt'
    burnc_dir = "/home/elizabeth/repos/kaldi/egs/burnc/kaldi_features/data"
    pros_feat_dir = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/burnc'
    mfcc_dir = '/home/elizabeth/repos/kaldi/egs/burnc/kaldi_features/data/train_breath_tok/feats.scp'
    kaldi_dir = 'tmp'
    save_dir = 'tmp'

    proc = BurncPreprocessor(burnc_dir,pros_feat_dir,mfcc_dir,kaldi_dir,speakers_file,save_dir)
    proc.preproc()

if __name__ == "__main__":
    main()


