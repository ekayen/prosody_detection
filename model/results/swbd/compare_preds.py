import pandas as pd
import pickle
from tabulate import tabulate


with open('../../../data/swbd_new_only/swbd_new_only.pkl','rb') as f:
    swbd = pickle.load(f)

sp_pred = 'is_speech_new_only_tenfold1_s256_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v7800.pred'
txt_pred = 'is_text_new_only_tenfold1_s256_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v7800.pred'
both_pred = 'is_both_new_only_tenfold1_s256_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v7800.pred'

"""
sp_pred = 'is_speech_new_heads_only_tenfold1_s256_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v5000.pred'
txt_pred = 'is_text_new_heads_only_tenfold1_s256_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v5000.pred'
both_pred = 'is_both_new_heads_only_tenfold1_s256_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v5000.pred'
"""


sp = pd.read_csv(sp_pred,sep='\t').rename(columns={"predicted_labels": "sp_pred"})
txt = pd.read_csv(txt_pred,sep='\t').rename(columns={"predicted_labels": "txt_pred"})
both = pd.read_csv(both_pred,sep='\t').rename(columns={"predicted_labels": "both_pred"})

assert(sum(list(sp['utt']==txt['utt']))==len(sp))
assert(sum(list(sp['utt']==both['utt']))==len(sp))

txt = txt.drop(['utt','labels','text'],axis=1)
both = both.drop(['utt','labels','text'],axis=1)

all_df = pd.concat([sp,txt,both],axis=1,join='inner')

txt_correct_mask = all_df['txt_pred']==all_df['labels']
sp_correct_mask = all_df['sp_pred']==all_df['labels']
both_correct_mask = all_df['both_pred']==all_df['labels']

txt_wrong_mask = all_df['txt_pred']!=all_df['labels']
sp_wrong_mask = all_df['sp_pred']!=all_df['labels']
both_wrong_mask = all_df['both_pred']!=all_df['labels']

txt_correct = all_df[txt_correct_mask]
sp_correct = all_df[sp_correct_mask]

txt_correct_sp_wrong = all_df[txt_correct_mask & sp_wrong_mask]
txt_wrong_sp_correct = all_df[txt_wrong_mask & sp_correct_mask]
both_correct_txt_wrong = all_df[both_correct_mask & txt_wrong_mask]
both_wrong_txt_correct = all_df[both_wrong_mask & txt_correct_mask]

def print_instances(df,filepath,print_pos=True,relevant_pos=None):
    with open(filepath,'w') as f:
        for i,row in df.iterrows():
            sent = row['text']
            lbls = row['labels']
            sp_pred = row['sp_pred']
            txt_pred = row['txt_pred']
            both_pred = row['both_pred']
            utt = row['utt']
            full_lbls = swbd['utt2new'][utt]
            pos = [swbd['tok2pos'][tok] for tok in swbd['utt2toks'][utt]]

            f.write(utt)
            f.write('\n')
            f.write(sent)
            f.write('\n')
            #if print_pos: f.write(f'{pos}\n')
            f.write(tabulate([['text']+sent.split(),
                              ['pos']+pos,
                              ['labels'] + lbls.split(),
                              ['full labels'] + full_lbls,
                              ['sp_pred']+sp_pred.split(),
                              ['txt_pred']+txt_pred.split(),
                              ['both_pred']+both_pred.split()], headers=[]))
            if relevant_pos:
                f.write(f'\n{" ".join(relevant_pos[i])}\n')
            """
            f.write('\n')
            f.write(f'labels\t\t\t {lbls}')
            f.write('\n')
            f.write(f'speech predictions \t {sp_pred}')
            f.write('\n')
            f.write(f'text predictions \t {txt_pred}')
            f.write('\n')
            f.write(f'both predictions \t {both_pred}')
            """
            f.write('\n')
            f.write('\n')


"""
txt_correct.to_csv('new_only_txt_correct.csv', sep='\t')
sp_correct.to_csv('new_only_sp_correct.csv', sep='\t')
txt_correct_sp_wrong.to_csv('new_only_txt_correct_sp_wrong.csv', sep='\t')
txt_wrong_sp_correct.to_csv('new_only_txt_wrong_sp_correct.csv', sep='\t')
both_correct_txt_wrong.to_csv('new_only_both_correct_txt_wrong.csv', sep='\t')
both_wrong_txt_correct.to_csv('new_only_both_wrong_txt_correct.csv', sep='\t')
"""
print_instances(txt_correct,'new_heads_only_txt_correct.csv')
print_instances(sp_correct,'new_heads_only_sp_correct.csv')
print_instances(txt_correct_sp_wrong,'new_heads_only_txt_correct_sp_wrong.csv')
print_instances(txt_wrong_sp_correct,'new_heads_only_txt_wrong_sp_correct.csv')
print_instances(both_correct_txt_wrong,'new_heads_only_both_correct_txt_wrong.csv')
print_instances(both_wrong_txt_correct,'new_heads_only_both_wrong_txt_correct.csv')

# Select instances based on pos tags

new_verbs_mask = [False for i in range(len(sp))]
new_nouns_mask = [False for i in range(len(sp))]
new_mods_mask = [False for i in range(len(sp))]
freq_pos_mask = [False for i in range(len(sp))]

nominal_pos = set(['^NN^NNS','PRP','PRP$','^PRP','^NN','NNPS','NNP','^NNP',
                   '^PRP$','NN', 'NNS','NNS^POS', '^NNS^POS','^DT^NN','^NNS',])
verbal_pos = set(['^VBZ','^VBP^RB','^VBP','VBD','^VBG','VBG','VBZ','VBP','^VBN','VBN','^VB','VB','^VB^RP','^VBD'])
mod_pos = set([ 'RB', '^JJ','JJ','RBR', 'JJS', 'JJR', 'RBS', 'JJ|RB', '^RB'])
freq_new_pos = set(['NN','DT','IN','JJ','NNS','RB','CD','CC','VB','VBP'])

for i,row in sp.iterrows():
    utt = row['utt']
    pos = [swbd['tok2pos'][tok] for tok in swbd['utt2toks'][utt]]
    newness = swbd['utt2new'][utt]
    for j in range(len(pos)):
        if newness[j] == 1:
            if not pos[j] in freq_new_pos:
                freq_pos_mask[i] = True
            if pos[j] in verbal_pos:
                new_verbs_mask[i] = True
            elif pos[j] in nominal_pos:
                new_nouns_mask[i] = True
            elif pos[j] in mod_pos:
                new_mods_mask[i] = True
            #import pdb;pdb.set_trace()

new_verbs_df = all_df[pd.Series(new_verbs_mask)]
new_nouns_df = all_df[pd.Series(new_nouns_mask)]
new_mods_df = all_df[pd.Series(new_mods_mask)]
infreq_new_df = all_df[pd.Series(freq_pos_mask)]

print_instances(new_verbs_df,'new_verbs.csv',print_pos=True)
print_instances(new_nouns_df,'new_nouns.csv',print_pos=True)
print_instances(new_mods_df,'new_mods.csv',print_pos=True)
print_instances(infreq_new_df, 'new_infreq.csv',print_pos=True)

# Look for cases where speech is right and text is wrong based on single words, not on the whole instance at once

both_correct_txt_wrong_mask = [False for i in range(len(sp))]
sp_correct_txt_wrong_mask = [False for i in range(len(sp))]
both_correct_txt_wrong_pos = []
sp_correct_txt_wrong_pos = []
sp_pos_freq = {}
both_pos_freq = {}
for i,row in sp.iterrows():
    both_pos = []
    sp_pos = []
    utt = row['utt']
    pos = [swbd['tok2pos'][tok] for tok in swbd['utt2toks'][utt]]
    both_pred = both.iloc[i]['both_pred'].split()
    txt_pred = txt.iloc[i]['txt_pred'].split()
    sp_pred = sp.iloc[i]['sp_pred'].split()
    labels = row['labels']
    for j in range(len(pos)):
        if labels[j] == both_pred[j] and not labels[j] == txt_pred[j] and labels[j] == '1':
            both_correct_txt_wrong_mask[i] = True
            both_pos.append(pos[j])
            if pos[j] in both_pos_freq:
                both_pos_freq[pos[j]] += 1
            else:
                both_pos_freq[pos[j]] = 1
        if labels[j] == sp_pred[j] and not labels[j] == txt_pred[j] and labels[j] == '1':
            sp_correct_txt_wrong_mask[i] = True
            sp_pos.append(pos[j])
            if pos[j] in sp_pos_freq:
                sp_pos_freq[pos[j]] += 1
            else:
                sp_pos_freq[pos[j]] = 1
    both_correct_txt_wrong_pos.append(both_pos)
    sp_correct_txt_wrong_pos.append(sp_pos)

both_correct_txt_wrong_by_tok = all_df[pd.Series(both_correct_txt_wrong_mask)]
sp_correct_txt_wrong_by_tok = all_df[pd.Series(sp_correct_txt_wrong_mask)]


print_instances(sp_correct_txt_wrong_by_tok,'sp_correct_txt_wrong_by_tok.csv',print_pos=True,relevant_pos=sp_correct_txt_wrong_pos)
print_instances(both_correct_txt_wrong_by_tok,'both_correct_txt_wrong_by_tok.csv',print_pos=True,relevant_pos=both_correct_txt_wrong_pos)

print_instances(all_df,'dev_predictions.csv',print_pos=True)

import pdb;pdb.set_trace()