# stars

Other things to maybe change:
* Brackets around unpronounced parts of words -- conveys important info, but also could mess with vocab size a lot.
* 'laughter' tokens -- same problem

Things I did change:
* leading and trailing hyphen on tokens -- dropped
* mispronunciations -- replaced with target tokens

Things I'm not going to change:
* tokens with '_1' on them (notation for common variants like 'cause' for 'because')


## Models

#### Speech model 

`model.py` is a CNN+RNN setup. 

`cnn_model.py` is the CNN with pooling and a fully-connected layer. 

The models prefixed with 'synth' are made to deal with the 1d synthetic data instead of the full MFCCs.

Currently all configuration is done inside the model file itself. No commandline args or config files needed.

#### Text model

`model.py` is a BiLSTM that performs binary token-level labeling. Config files are found in the corresponding subdir.
