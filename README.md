# stars

#### NXT data preprocessing notes:

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


The models prefixed with 'synth' are made to deal with the 1d synthetic data instead of the full MFCCs.

#### Text model

`model.py` is a BiLSTM that performs binary token-level labeling. Config files are found in the corresponding subdir.
