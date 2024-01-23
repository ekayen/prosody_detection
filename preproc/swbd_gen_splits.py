import math
import yaml
import pickle
import random
import os

data_path = '../data/swbd'
#data_path = '../data/swbd_new_only'
dict_file = 'swbd.pkl'
#dict_file = 'swbd_kontrast.pkl'
#dict_file = 'swbd.unsplit.nps.pkl'
#dict_file = 'swbd_acc.pkl'


with open(os.path.join(data_path,dict_file),'rb') as f:
    nested = pickle.load(f)

utt_ids = list(set([utt for para in nested for utt in nested['utt2toks']]))

print(f'Total utts: {len(utt_ids)}')

#WRITE_OUTPUT = True
WRITE_OUTPUT = False
output = 'tone_marked'
# output = 'all'

if output == 'tone_marked':
    ############### only use tone-marked utt ids:
    tone_marked_utts = []
    tone_marked_toks = list(nested['tok2tone'].keys())
    tone_marked_utts = list(set([nested['tok2utt'][tok] for tok in tone_marked_toks if tok in nested['tok2utt']]))

    # Some of the toks still aren't in the tok2tone dict, so we'll take out those utts by hand:
    remove_utts = set()
    for utt in tone_marked_utts:
        if not utt in nested['utt2toks']:
            remove_utts.add(utt)
        for tok in nested['utt2toks'][utt]:
            if not tok in nested['tok2tone']:
                remove_utts.add(utt)
    for utt in remove_utts:            
        tone_marked_utts.remove(utt)

    print('Total tone marked utts:',len(tone_marked_utts))

    utt_ids = tone_marked_utts

def gen_pos_vocab(split_dict):
    pos2freq = {}
    labels = set()
    for utt_id in split_dict['train']:
        tokens = nested['utt2toks'][utt_id]
        pos_strs = [nested['tok2pos'][tok] for tok in tokens]

        for pos in pos_strs:
            if pos in pos2freq:
                pos2freq[pos] += 1
            else:
                pos2freq[pos] = 1
    i2lbl = {0:'PAD'}
    lbl2i = {'PAD':0}
    for i,lbl in enumerate(labels):
       lbl2i[lbl] = i + 1
       i2lbl[i+1] = lbl

    ordered_pos = [key for key,value in sorted(pos2freq.items(), key=lambda item: item[1],reverse=True)]
    pos2i = {'PAD':0,
             'UNK':1}
    i2pos = {0:'PAD',
             1:'UNK'}
    for i,pos in enumerate(ordered_pos):
        pos2i[pos] = i + 2
        i2pos[i + 2] = pos
    pos_vocab_dict = {'w2i':pos2i,
                      'i2w':i2pos,
                      'w2freq':pos2freq,
                      'i2lbl':i2lbl,
                      'lbl2i':lbl2i}
    #return w2i,i2w,w2freq
    return pos_vocab_dict


def gen_vocab(split_dict):
    w2freq = {}
    labels = set()
    for utt_id in split_dict['train']:
        tokens = nested['utt2toks'][utt_id]
        tok_strs = [nested['tok2str'][tok] for tok in tokens]


        for tok in tok_strs:
            if tok in w2freq:
                w2freq[tok] += 1
            else:
                w2freq[tok] = 1
        #TODO: add something like:
        #import pdb;pdb.set_trace()
        #labels.update(set(nested['utt2bio'][utt_id]))

    i2lbl = {0:'PAD'}
    lbl2i = {'PAD':0}
    for i,lbl in enumerate(labels):
       lbl2i[lbl] = i + 1
       i2lbl[i+1] = lbl


    ordered_toks = [key for key,value in sorted(w2freq.items(), key=lambda item: item[1],reverse=True)]
    w2i = {'PAD':0,
           'UNK':1}
    i2w = {0:'PAD',
           1:'UNK'}
    for i,tok in enumerate(ordered_toks):
        w2i[tok] = i + 2
        i2w[i + 2] = tok
    vocab_dict = {'w2i':w2i,
                  'i2w':i2w,
                  'w2freq':w2freq,
                  'i2lbl':i2lbl,
                  'lbl2i':lbl2i}
    #return w2i,i2w,w2freq
    print('vocab size:',len(vocab_dict['w2i']) - 2)
    return vocab_dict

##################################################################
# First set of splits:
# totally random, no consideration of speaker
# 10 splits, 0 is default
##################################################################
#num_folds = 5
num_folds = 10

seeds = [860, 33, 616, 567, 375, 262, 293, 502, 295, 886]
utt_ids = sorted(utt_ids)
random.seed(seeds[0])
random.shuffle(utt_ids)
test_start = 0

if num_folds==10:
    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1
    output_name = 'tenfold'
    if output == 'tone_marked':
        output_name = output_name + '_' + output
elif num_folds==5:
    train_ratio = 0.6
    dev_ratio = 0.2
    test_ratio = 0.2
    output_name = 'fivefold'
elif num_folds==1:
    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1
    output_name = 'swbd.unsplit.nps'

    #for i,seed in enumerate(seeds):
for i in range(num_folds):
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

    vocab_dict = gen_vocab(split_ids)
    pos_vocab_dict = gen_pos_vocab(split_ids)

    if WRITE_OUTPUT:
        with open(os.path.join(data_path,'splits',f'{output_name}{i}.yaml'), 'w') as f:
            yaml.dump(split_ids, f)
        with open(os.path.join(data_path,'splits',f'{output_name}{i}.vocab'),'wb') as f:
            pickle.dump(vocab_dict, f)
        with open(os.path.join(data_path,'splits',f'{output_name}{i}.posvocab'),'wb') as f:
            pickle.dump(pos_vocab_dict, f)

"""        

total_utts = len(tone_marked_utts)
train_end = math.floor(total_utts*0.8)
dev_end = math.floor(total_utts*0.9)
print('train len:',train_end)
print('dev len:',dev_end - train_end)
print('test len:',total_utts - dev_end)

split_ids = {'train':tone_marked_utts[:train_end],
             'dev':tone_marked_utts[train_end:dev_end],
             'test':tone_marked_utts[dev_end]}

vocab_dict = gen_vocab(split_ids)
pos_vocab_dict = gen_pos_vocab(split_ids)

with open(os.path.join(data_path,'splits',f'swbd_tone.yaml'), 'w') as f:
    yaml.dump(split_ids, f)
with open(os.path.join(data_path,'splits',f'swbd_tone.vocab'),'wb') as f:
    pickle.dump(vocab_dict, f)
with open(os.path.join(data_path,'splits',f'swbd_tone.posvocab'),'wb') as f:
    pickle.dump(pos_vocab_dict, f)


import pdb;pdb.set_trace()    
"""
