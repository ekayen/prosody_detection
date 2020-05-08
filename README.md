# Prosody detection

This repo contains code for a CNN+LSTM-based pitch accent detection model for English.

##Preprocessing

###Feature extraction

We use (OpenSMILE)[https://www.audeering.com/opensmile/] for feature extraction. 

1. Download OpenSMILE and follow installation instructions.

2. Download (BURNC)[https://catalog.ldc.upenn.edu/LDC96S36] (or another speech corpus). If the audio files are not already in wav format, convert them to wav using (sox)[https://linux.die.net/man/1/sox]. This README assumpes you also copy all wav files to subdir of the OpenSMILE dir called `burnc`: e.g., `~/opensmile-2.3.0/burnc/`. If you leave the audio files in the original BURNC file structure or some other arrangement, you will need to edit the filepath in the `burnc_feat_extract.sh` script.

3. Copy the config files from the `opensmile` subdir or this repo to the config subdir of the OpenSMILE directory (assuming you downloaded OpenSMILE to `~`):

`cp preproc/opensmile/IS13* ~/opensmile-2.3.0/config/`

4. Copy the feature extraction script to the OpenSMILE root dir:

`cp preproc/opensmile/burnc_feat_extract.sh ~/opensmile-2.3.0/

5. Go to the OpenSMILE directory and run feature extraction:

```
cd ~/opensmile-2.3.0
./burnc_feat_extract.sh```



###Preparing model input

##Training and evaulation