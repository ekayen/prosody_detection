#!/usr/bin/env python
# coding: utf-8

# # Data preparation
# 
# #### Current format: 
# Tokens each have an xml entry in their phonwords file. 
# Accents each have an xml entry in their accents file.
# Turns have are in their own files, and are recorded as time spans. The time span of a phonword is somewhere inside the time span of a turn.
# 
# Note: Each file is per-speaker, not per-turn or per-conversation.
# 
# #### Desired format:
# One turn per line, whitespace-separated tokens, tab, binary string with 1 for nuclear pitch accent.

import os
import numpy as np
import xml.etree.ElementTree as ET
import pickle

# Find files:
data_dir = '/afs/inf.ed.ac.uk/group/corpora/large/switchboard/nxt/xml'
#data_dir = '/home/ekayen/repos/stars/nxt-subset'

mispron_dict = {
    '[row/wow]':'wow',
    '[mot/lot]':'lot',
    '[trest/test]':'test',
    '[adamnet/adapted]':'adapted',
    '[storly/story]':'story',
    '[unconvenient/inconvenient]':'inconvenient',
    '[dib/bit]': 'bit',
    '[tack/talking]': 'talking',
    '[banding/banning]': 'banning',
    '[ruther/rather]': 'rather',
    '[shrip/shrimp]': 'shrimp',
}

NUC_ONLY = True # if true, only consider nuclear accents; if false, consider all accents
if NUC_ONLY:
    out_pickle = 'nuc_only.pickle'
    out_txt = 'nuc_only.txt'
    out_vocab = 'nuc_vocab.pickle'
else:
    out_pickle = 'all_acc.pickle'
    out_txt = 'all_acc.txt'
    out_vocab = 'all_vocab.pickle'

wd_to_i = {}
i_to_wd = {}
id_to_acc = {}
counter = 0
users = ('A','B')
lines = []

nite = '{http://nite.sourceforge.net/}'

accent_dict = {'nuclear':1,
               'plain':0,
               'pre-nuclear':0}



dialog_nums = list(set([f.split('.')[0] for f in os.listdir(os.path.join(data_dir,'accent'))]))

if NUC_ONLY:
    dialog_nums_tmp = []
    for dial in dialog_nums:
        with open(os.path.join(data_dir,'accent','.'.join([dial,'A','accents','xml'])),'r') as f:
            txt = f.read()
            if 'type="' in txt:
                dialog_nums_tmp.append(dial)
    dialog_nums=dialog_nums_tmp


for dialog_num in dialog_nums:
    #print('processing ',dialog_num,'...')

    turn_files = [os.path.join(data_dir,'turns','.'.join([dialog_num,user,'turns','xml'])) for user in users]
    acc_files = [os.path.join(data_dir,'accent','.'.join([dialog_num,user,'accents','xml'])) for user in users]
    wd_files = [os.path.join(data_dir,'phonwords','.'.join([dialog_num,user,'phonwords','xml'])) for user in users]

    words = []
    ids = []
    times = []

    for i,wd_file in enumerate(wd_files):
        tmp_wds = []
        tmp_ids = []
        tmp_times = []
        wd_tree = ET.parse(wd_file)
        wd_root = wd_tree.getroot()
        for phonword in wd_root.findall('phonword'):
            orth = phonword.attrib['orth']
            # Clean up orth form:
            orth = orth.strip('-')
            orth = orth.strip('{')
            orth = orth.strip('}')
            if orth in mispron_dict:
                print(orth)
                orth = mispron_dict[orth]
            id_num = phonword.attrib[nite+'id']
            start_time = float(phonword.attrib[nite+'start'])
            if not orth in wd_to_i:
                wd_to_i[orth] = counter
                i_to_wd[counter] = orth
                counter += 1
            tmp_wds.append(wd_to_i[orth])
            tmp_ids.append(id_num) # TODO since these ids are unique, I can make a lookup table for them too for speed
            tmp_times.append(start_time)
        words.append(tmp_wds)
        ids.append(tmp_ids)
        times.append(tmp_times)

        acc_tree = ET.parse(acc_files[i])
        acc_root = acc_tree.getroot()
        for acc in acc_root.findall('accent'):
            for chld in acc:
                acc_id = chld.attrib['href'].split('(')[-1][:-1]
                if NUC_ONLY:
                    id_to_acc[acc_id] = accent_dict[acc.attrib['type']]
                else:
                    id_to_acc[acc_id] = 1

    # ### Make np array of accents

    words = (np.array(words[0]), np.array(words[1]))
    times = (np.array(times[0]), np.array(times[1]))

    accents = (np.zeros(words[0].shape,dtype=np.int32),np.zeros(words[1].shape,dtype=np.int32))

    for i in (0,1):
        for j in range(words[i].shape[0]):
            id_num = ids[i][j]
            if id_num in id_to_acc:
                accents[i][j] = id_to_acc[id_num]
            else:
                accents[i][j] = 0


    # ### Iterate through turns, writing the final form out turn by turn

    turns0 = [(float(child.attrib[nite+'id'][1:].replace('-','.')), (float(child.attrib[nite+'start']), float(child.attrib[nite+'end']),0)) for child in ET.parse(turn_files[0]).getroot() if nite+'start' in child.attrib]
    turns1 = [(float(child.attrib[nite+'id'][1:].replace('-','.')), (float(child.attrib[nite+'start']), float(child.attrib[nite+'end']),1)) for child in ET.parse(turn_files[1]).getroot() if nite+'start' in child.attrib]
    turns = turns0 + turns1
    turns.sort()
    turns = dict(turns)

    for num in turns:
        speaker = turns[num][2]
        start = turns[num][0]
        end = turns[num][1]
        mask = np.squeeze(np.logical_and([times[speaker] >= start], [times[speaker] <= end]))
        turn = words[speaker][mask]
        acc = accents[speaker][mask]
        tokens = ' '.join([i_to_wd[i] for i in turn])
        labels = ' '.join([str(i) for i in acc])
        lines.append((tokens,labels))

# Pickle the results
with open(out_pickle,'wb') as f:
    pickle.dump(lines,f)

with open(out_vocab,'wb') as f:
    pickle.dump((wd_to_i,i_to_wd),f)

# Also write them to a text file:
with open(out_txt,'w') as f:
    for line in lines:
        f.write(line[0]+'\t'+str(line[1])+'\n')




