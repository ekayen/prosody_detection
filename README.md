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

Sequence labelling -- LSTM-CRF is popular, but it's a char+word level model, and I don't think I want this thing doing char level. It also presumably takes more data cause it's heftier._1
(Here's a library, in case: https://arxiv.org/pdf/1806.05626.pdf)

For a baseline, would prefer to do a bland bilstm (keras tutorial for sanity: https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/)