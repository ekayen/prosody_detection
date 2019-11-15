# Preprocessing speech features.

## Feature extraction:

Because of the particular way Kaldi features are generated, the preprocessing procedure is quite different depending on which feature extraction method you use. In the case of Kaldi, a lot of the text processing has to be done before feature extraction; in OpenSMILE, the text processing can happen at any point relative to speech feature extraction.

1. Kaldi feature extraction

Start by installing Kaldi according to the instructions on the Kaldi website

#### For BURNC:

In order to do Kaldi feature extraction, you have to first get all of the data into a Kaldi-amenable format. First, you'll need a subdirectory to do your work in within the Kaldi repo:

`kaldi/egs/burnc`

I recommend creating this subdir by copying the structure of the SWBD recipe, including the data, utils, and steps subdirs (the latter two should be symlinks). Put the actual data from BURNC into the data subdir, or adjust all filepaths in subsequent steps to point to wherever BURNC is located.

First, you need to generate a certain number of files at the path `data/train` inside your subdir:

- utt2spk: each line has two strings: the key of an utterance and the identifier for the utterance's speaker
 `<utt_key> <spk_id>`
- text: each line has the key to the utterance, followed by the text string of the utterance. Everything is separated by spaces, not tabs.
`<utt_key> <text>`
- wav.scp: each line has the key to a recording and then the filepath to the actual location of that recording.
`<recording_id> <recording_path>
- segments: each line has the utterance, the recording it comes from, and a start and an end time

These are all produced and placed in the appropriate file by the script `burnc_data_prep.py`. This script also generates the following files that are used by the model (though not by Kaldi):

- text2labels: each line has the key to the utterance, the text of the utterance, and the list of labels. Each field is tab-separated; within each field, tokens are space-separated.
- utt2toktime: each line has the key to the utterance and then the list of start times for each token. It also has the end time for the final token. Again, fields are tab-separated, and internal divisions within each field are space-separated.

Currently, both text2labels and utt2toktime are generated at the path `kaldi/egs/burnc/kaldi_features/data/train` (or wherever else might be specified in `paths.py`), but need to be manually copied over to stars/data/burnc

Once all these files are generated, you just need to generate the file `spk2utt` that is provided by kaldi (NOTE: the path for the `utt2spk` and `spk2utt` files should match the path you have in `paths.py`.)

`./utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt`

Before generating anything, set all your paths by calling the following:

```
source cmd.sh
source path.sh
```

Then you can generate the mfcc+pitch features (NOTE: again, make sure that the paths shown here as `data/train` and `mfcc_pitch` are configured correctly. Also, the `--nj` argument specifies the number of cpus to run on -- check how many your machine has to work with and set accordingly.0

`.steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc.conf --nj 6 data/train/ exp/make_mfcc_pitch mfcc_pitch`

You can then apply Cepstral Mean Variance Normalization:

```
./steps/compute_cmvn_stats.sh data/train exp/make_mfcc_pitch/train mfcc_pitch
./local/cmvn.sh --cmd "$train_cmd" data/train exp/make_cmvn_dd/train mfcc_pitch

```

#### For SWBD-NXT

There is an existing recipe for SWBD (which SWBD-NXT is a subset of), found at `kaldi/egs/swbd/s5c`. 

This means you can start directly with feature extraction (which takes hours, fair warning):

```
source cmd.sh
source path.sh
.steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc.conf --nj 6 data/train/ exp/make_mfcc_pitch mfcc_pitch
./steps/compute_cmvn_stats.sh data/train exp/make_mfcc_pitch/train mfcc_pitch
./local/cmvn.sh --cmd "$train_cmd" data/train exp/make_cmvn_dd/train mfcc_pitch

```
(Note: I'm not sure if that last `cmvn.sh` script exists in the SWBD recipe? But it's just calling `apply_cmvn` with the appropriate arguments.)

Once you've done this, you still need to do the preprocessing of text, which you can do by calling `prep_nxt_text.py`


2. OpenSMILE feature extraction

OpenSMILE feature extraction doesn't require anything but the audio files as input. You'll segment the audio out later. To do the extraction, you need all your audio files in wav format in a directory.

Then, using the config file that you'll find here once I'm done making it, call openSMILE on each wav file:

./SMILExtract -C <path to config> -I <input wav file> -O <output csv file>


## Prepping data for the model:


Most of the text preprocessing is already done at this point. Just make sure you've copied over text2labels and utt2toktimes to the stars data directory (in the case of BURNC), and then run either `filter_feats_burnc.ipynb` or `filter_feats.ipynb` (for SWBD data). If you've applied cmvn to SWBD and consequently have the features in text format, then use `filter_cmvn.ipynb` instead. These scripts will load the speech and text features into python dictionaries and then pickle them to make them easy for the model to subsequently use. In the case of SWBD, this step also drops a huge segment of the overall data