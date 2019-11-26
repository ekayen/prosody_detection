from burnc_data_prep import gen_kaldi_inputs
from paths import burnc_dir,kaldi_dir
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
    gen_kaldi_inputs(burnc_dir,kaldi_dir,speakers_file)


if __name__ == "__main__":
    main()