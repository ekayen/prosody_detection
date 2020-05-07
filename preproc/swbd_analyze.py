import pickle

swbd_file = '../data/swbd/swbd.pkl'

with open(swbd_file,'rb') as f:
    swbd = pickle.load(f)

utt2toks = swbd['utt2toks']
utt2frames = swbd['utt2frames']


"""
tok_lens = []
frame_lens = []
for utt in utt2toks:
    tok_lens.append(len(utt2toks[utt]))
    #import pdb;pdb.set_trace()
    frame_len = utt2frames[utt][-1].item()
    if frame_len == 0:
        import pdb;pdb.set_trace()
    frame_lens.append(frame_len)


max_toks = max(tok_lens)
min_toks = min(tok_lens)
avg_toks = sum(tok_lens)/len(tok_lens)

print(f'max toks: {max_toks}')
print(f'min toks: {min_toks}')
print(f'avg toks: {avg_toks}')
    

max_frame = max(frame_lens)
min_frame = min(frame_lens)
avg_frame = sum(frame_lens)/len(frame_lens)

print(f'max frames: {max_frame}')
print(f'min frames: {min_frame}')
print(f'avg frames: {avg_frame}')
"""
news = 0
olds = 0
meds = 0
other = 0

tok2infostat = swbd['tok2infostat']

import yaml
split = '../data/swbd/splits/tenfold1.yaml'
with open(split,'r') as f:
    split_dict = yaml.load(f)

trainset = split_dict['train']

other_lbls = set()

#for utt in utt2toks:
for utt in trainset:
    toks = utt2toks[utt]
    infostats = [tok2infostat[tok] for tok in toks]
    for inf in infostats:
        if inf == 'new':
            news += 1
        elif inf == 'old':
            olds += 1
        elif inf == 'med':
            meds += 1
        else:
            other += 1
            other_lbls.add(inf)

print(f'Total new: {news}, {100*(news/(news+olds+meds+other))}%')
print(f'Total old: {olds}, {100*(olds/(news+olds+meds+other))}%')
print(f'Total med: {meds}, {100*(meds/(news+olds+meds+other))}%')
print(f'Total other: {other}, {100*(other/(news+olds+meds+other))}%')
print(other_lbls)
print('-'*50)

utt2new = swbd['utt2new']

new_only = 0
not_new = 0
for utt in trainset:
    new = utt2new[utt]
    new_only += sum(new)
    not_new += len(new) - sum(new)

print(f'Total new: {new_only}, {100*(new_only/(new_only+not_new))}%')
print(f'Total non-new: {not_new}, {100*(not_new/(new_only+not_new))}%')
print('-'*50)

utt2old = swbd['utt2old']

old_only = 0
not_old = 0
for utt in trainset:
    old = utt2old[utt]
    old_only += sum(old)
    not_old += len(old) - sum(old)

print(f'Total old: {old_only}, {100*(old_only/(old_only+not_old))}%')
print(f'Total non-old: {not_old}, {100*(not_old/(old_only+not_old))}%')
print('-'*50)

utts_w_new = {}
for utt in utt2toks:
    news = sum(utt2new[utt])
    if news > 0:
        utts_w_new[utt] = utt2toks[utt]
        new = utt2new[utt]
        new_only += sum(new)
        not_new += len(new) - sum(new)

print(f'Total new in balanced-er set: {new_only}, {100*(new_only/(new_only+not_new))}%')
print(f'Total non-new in balanced-er set: {not_new}, {100*(not_new/(new_only+not_new))}%')
print('-'*50)

swbd_new_only = swbd
swbd_new_only['utt2toks'] = utts_w_new
swbd_new_only['utt_ids'] = list(utts_w_new.keys())
with open('../data/swbd_new_only/swbd_new_only.pkl','wb') as f:
    pickle.dump(swbd_new_only,f)


new_acc = 0
nonnew_acc = 0
new_nonacc = 0
nonnew_nonacc = 0
for utt in swbd['utt2toks']:
    for tok in swbd['utt2toks'][utt]:
        if tok in swbd['tok2tone']:
            infostat = swbd['tok2infostat'][tok]
            accent = swbd['tok2tone'][tok]
            if infostat=='new':
                if accent==1:
                    new_acc += 1
                elif accent==0:
                    new_nonacc +=1
            else:
                if accent==1:
                    nonnew_acc += 1
                elif accent==0:
                    nonnew_nonacc +=1

print(f'\tNew\tNonnew')
print(f'Acc\t{new_acc}\t{nonnew_acc}')
print(f'Nonacc\t{new_nonacc}\t{nonnew_nonacc}')

old_acc = 0
nonold_acc = 0
old_nonacc = 0
nonold_nonacc = 0
for utt in swbd['utt2toks']:
    for tok in swbd['utt2toks'][utt]:
        if tok in swbd['tok2tone']:
            infostat = swbd['tok2infostat'][tok]
            accent = swbd['tok2tone'][tok]
            if infostat=='old':
                if accent==1:
                    old_acc += 1
                elif accent==0:
                    old_nonacc +=1
            else:
                if accent==1:
                    nonold_acc += 1
                elif accent==0:
                    nonold_nonacc +=1

print(f'\tNonold\tOld')
print(f'Acc\t{nonold_acc}\t{old_acc}')
print(f'Nonacc\t{nonold_nonacc}\t{old_nonacc}')


#nominal_pos = set(['^NN^NNS','PRP','PRP$','^PRP','^NN','NNPS','NNP','^NNP',
#                   '^PRP$','NN', 'NNS','NNS^POS', '^NNS^POS','^DT^NN','^NNS'])
nominal_pos = set(['^NN^NNS','PRP','PRP$','^PRP','^NN','NNPS','NNP','^NNP',
                   '^PRP$','NN', 'NNS','NNS^POS', '^NNS^POS','^DT^NN','^NNS',
                   'JJ|RB', 'JJS', 'JJR','^JJ','JJ'])
verbal_pos = set(['^VBZ', '^VBP^RB', '^VBP', 'VBD','^VBG','VBG','VBZ', 'VBP',
                  '^VBN','VBN', '^VB', 'VB','^VB^RP','^VBD'])
#mod_pos = set([ 'RB', '^JJ','JJ','RBR', 'JJS', 'JJR', 'RBS', 'JJ|RB', '^RB'])

new_nom = 0
nonnew_nom = 0
new_nonnom = 0
nonnew_nonnom = 0
other_new = 0
other_nonnew = 0
for utt in swbd['utt2toks']:
    for tok in swbd['utt2toks'][utt]:
        newness = swbd['tok2infostat'][tok]
        pos = swbd['tok2pos'][tok]
        if pos in nominal_pos:
            if newness == 'new':
                new_nom += 1
            else:
                nonnew_nom += 1
        elif pos in verbal_pos:# or pos in mod_pos:
            if newness == 'new':
                new_nonnom += 1
            else:
                nonnew_nonnom += 1
        else:
            if newness == 'new':
                other_new += 1
            else:
                other_nonnew += 1



print(f'\tNew\tNonnew')
print(f'Nom\t{new_nom}\t{nonnew_nom}')
print(f'Vrb\t{new_nonnom}\t{nonnew_nonnom}')
print(f'Other\t{other_new}\t{other_nonnew}')


new_vrb = 0
nonnew_vrb = 0
new_nonvrb = 0
nonnew_nonvrb = 0
for utt in swbd['utt2toks']:
    for tok in swbd['utt2toks'][utt]:
        newness = swbd['tok2infostat'][tok]
        pos = swbd['tok2pos'][tok]
        if pos in verbal_pos:
            if newness == 'new':
                new_vrb += 1
            else:
                nonnew_vrb += 1
        elif pos in nominal_pos:# or pos in mod_pos:
            if newness == 'new':
                new_nonvrb += 1
            else:
                nonnew_nonvrb += 1



print(f'\tNew\tNonnew')
print(f'Vrb\t{new_vrb}\t{nonnew_vrb}')
print(f'Nonvrb\t{new_nonvrb}\t{nonnew_nonvrb}')

# Count number of
pos2new_count = {}
pos2count = {}
for utt in swbd['utt2toks']:
    for tok in swbd['utt2toks'][utt]:
        newness = swbd['tok2infostat'][tok]
        pos = swbd['tok2pos'][tok]
        if pos in pos2count:
            pos2count[pos] += 1
        else:
            pos2count[pos] = 1
        if newness == 'new':
            if pos in pos2new_count:
                pos2new_count[pos] += 1
            else:
                pos2new_count[pos] = 1

pos2percent = {}
for pos in pos2count:
    if pos in pos2new_count:
        pos2percent[pos] = pos2new_count[pos]/pos2count[pos]
    else:
        pos2percent[pos] = 0

print('-'*50)
filename = 'swbd_pos_newness_stats.tmp'
sorted_pos = [k for k, v in sorted(pos2percent.items(), key=lambda item: item[1],reverse=True)]
counter = 50
with open(filename,'w') as f:
    for i,pos in enumerate(sorted_pos):
        f.write(f'{pos}\t{pos2percent[pos]}\n')
    total = sum(pos2new_count.values())
    f.write(f'Total\t{total}')



import pdb;pdb.set_trace()