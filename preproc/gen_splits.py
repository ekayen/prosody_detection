import yaml
import pickle
import random
import os

data_path = '../data/burnc'
dict_file = 'burnc.pkl'
train_ratio = 0.6
dev_ratio = 0.2
test_ratio = 0.2

with open(os.path.join(data_path,dict_file),'rb') as f:
    nested = pickle.load(f)

utt_ids = [utt for para in nested for utt in nested[para]['utterances']]

##################################################################
# First set of splits:
# totally random, no consideration of speaker
# 10 splits, 0 is default
##################################################################

#seeds = [860, 33, 616, 567, 375, 262, 293, 502, 295, 886]
seeds = [860, 33, 616, 567, 375]#, 262, 293, 502, 295, 886]
utt_ids = sorted(utt_ids)
random.seed(seeds[0])
random.shuffle(utt_ids)
test_start = 0

#for i,seed in enumerate(seeds):
for i in range(5):
    test_end = int(round(test_start + test_ratio* len(utt_ids)))
    print(test_end)
    split_ids = test_start
    split_ids = {}
    split_ids['test'] = utt_ids[test_start:test_end]
    other_ids = utt_ids[:test_start]+utt_ids[test_end:]
    random.seed(seeds[i])
    random.shuffle(other_ids)
    dev_idx = int(round(dev_ratio*len(utt_ids)))
    split_ids['dev'] = other_ids[:dev_idx]
    split_ids['train'] = other_ids[dev_idx:]

    test_start = test_end
    with open(os.path.join(data_path,'splits',f'fivefold{i}.yaml'), 'w') as f:
        yaml.dump(split_ids, f)


##################################################################
# Second set of splits:
# each speaker held out, with held out size of 20%
##################################################################
with open('burnc_speakers.txt','r') as f:
    speakers = [line.strip() for line in f.readlines()]

traindev = []
devset_size = int(len(utt_ids)*0.2)
print(f'devset_size: {devset_size}')
for speaker in speakers:
    gender = speaker[0]
    split_ids = {'train': [],
                 'dev': [],
                 'test': []}
    utt_ids = sorted(utt_ids)
    for utt_id in utt_ids:
        if not utt_id.startswith(speaker):
            traindev.append(utt_id)
        else:
            split_ids['test'].append(utt_id)

    random.seed(531)
    random.shuffle(traindev)
    devset = 0
    for ex in traindev:
        if ex.startswith(gender) and devset <= devset_size:
            split_ids['dev'].append(ex)
            devset += 1
        else:
            split_ids['train'].append(ex)
    with open(os.path.join(data_path, 'splits', f'heldout_{speaker}.yaml'), 'w') as f:
        yaml.dump(split_ids, f)

##################################################################
# Third set of splits:
# f2b only
##################################################################
train_ratio = 0.6
dev_ratio = 0.2
f2b_utts = [utt_id for utt_id in utt_ids if utt_id.startswith('f2b')]
train_idx = int(len(f2b_utts)*train_ratio)
dev_idx = int(len(f2b_utts)*(train_ratio+dev_ratio))
seeds = [122, 451, 917, 798, 170, 528, 337, 25, 195, 564]
for i,seed in enumerate(seeds):
    f2b_utts = sorted(f2b_utts)
    random.seed(seed)
    random.shuffle(f2b_utts)
    split_ids['train'] = utt_ids[:train_idx]
    split_ids['dev'] = utt_ids[train_idx:dev_idx]
    split_ids['test'] = utt_ids[dev_idx:]
    with open(os.path.join(data_path, 'splits', f'f2b_only{i}.yaml'), 'w') as f:
        yaml.dump(split_ids, f)

