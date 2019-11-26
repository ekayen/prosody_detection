from burnc_preproc import BurncPreprocessor
from paths import burnc_dir,kaldi_dir,speakers_file
"""
    {paragraph_id:
        {utterances:
            {
                utt_id1: [tok_id1, tok_id2, ...],
                utt_id2: [tok_id8, tok_id9, ...]
            }
        },
        {token_breaks:
            {
                tok_id1: (start, end),
                tok_id2: (start, end)
            }
        },
        {mfccp:
             {tok_id1: [feats],
              tok_id2: [feats]
              },
         },
        {prosodic:
             {tok_id1: [feats],
              tok_id2: [feats]
              },
         }
        {token_labels:
             {tok_id1: 0
              tok_id2: 1
                  ...
              }
         },
        {tokens:
             {tok_id1: ‘the’,
              tok_id2: ‘supreme’
        ...
        }
    }
}
"""


def gen_feat_dict(source='burnc',feats='pros'):
    '''
    :param source: corpus to use ('burnc' or 'swbd')
    :return: a nested dictionary with the format shown above that has all necessary features and info.
    '''
    paragraphs = []


def main():
    speakers_file = 'burnc_speakers.txt'
    burnc_dir = "/home/elizabeth/repos/kaldi/egs/burnc/kaldi_features/data"
    pros_feat_dir = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/burnc'
    mfcc_dir = '/home/elizabeth/repos/kaldi/egs/burnc/kaldi_features/data/train_breath_tok/feats.scp'
    out_dir = 'tmp'

    proc = BurncPreprocessor(burnc_dir, out_dir, pros_feat_dir, mfcc_dir, speakers_file)
    proc.text_preproc()

if __name__ == "__main__":
    main()


