# Prosody detection

This repo contains code for a CNN+LSTM-based pitch accent detection model for English.

## Preprocessing

### Feature extraction

We use [OpenSMILE](https://www.audeering.com/opensmile/) for feature extraction. We only need 6 prosodic features extracted, but we use the Interspeech 2013 feature set as a starting place and so end up extracting more features and discarding the irrelevant ones later in preprocessing.

1. Download OpenSMILE and follow installation instructions.

2. Download [BURNC](https://catalog.ldc.upenn.edu/LDC96S36) (or another speech corpus). If the audio files are not already in wav format, convert them to wav using [sox](https://linux.die.net/man/1/sox). These instructions assume you also copy all wav files to subdir of the OpenSMILE dir called `burnc`: e.g., `~/opensmile-2.3.0/burnc/`. If you leave the audio files in the original BURNC file structure or some other arrangement, you will need to edit the filepath in the `burnc_feat_extract.sh` script.

3. Copy the config files from the `opensmile` subdir or this repo to the config subdir of the OpenSMILE directory (assuming you downloaded OpenSMILE to `~`):

`cp preproc/opensmile/IS13* ~/opensmile-2.3.0/config/`

4. Copy the feature extraction script to the OpenSMILE root dir:

`cp preproc/opensmile/burnc_feat_extract.sh ~/opensmile-2.3.0/`

5. Go to the OpenSMILE directory and run feature extraction:

```
cd ~/opensmile-2.3.0
./burnc_feat_extract.sh
```

You may need to adjust the paths in the main function of `burnc_preproc.py` to point to the places where you have saved the raw BURNC data and extracted features. 

This same process can be adapted to preprocess Switchboard NXT as well, substituting the relevant SWBD scripts for BURNC scripts where necessary.

### Preparing model input

In this step, we will use the raw corpus data and the extracted features to generate a nested python dictionary containing all corpus instances and dictionaries recording mappings between IDs and data. We'll save this as a pickle file that we'll load into memory as part of the training and evaluation step.

```
cd ~/prosody_detection/preproc
./preprocess.sh
```

All preprocessing steps can also be done for SWBD NXT using the `preprocess_swbd.sh` script.

## Training and evaulation

To train on a given training split of the data and evaluate on the corresponding dev split, simply do the following:

```
cd ../model
python3 train.py -c conf/cnn_lstm_best.yaml
```

You can specify any config file as input. 