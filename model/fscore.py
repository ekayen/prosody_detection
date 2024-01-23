import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

output = 'seg_output/swbd_seg_w_interstices_train_s0_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v14555.pred'

df = pd.read_csv(output,sep='\t')


gold_pos = 0
true_pos = 0
false_pos = 0

all_gold = []
all_pred = []
for index,row in df.iterrows():
    gold = np.array([int(lbl) for lbl in row['labels'].split()])
    all_gold.append(gold)
    gold_pos += sum(gold)

    pred = np.array([int(lbl) for lbl in row['predicted_labels'].split()])
    pos_mask = pred == 1
    true_mask = gold == 1
    true_pos += np.sum(pos_mask & (true_mask==pos_mask))
    false_pos += np.sum(pos_mask & (true_mask!= pos_mask))
    all_pred.append(pred)

all_gold = np.concatenate(all_gold)
all_pred = np.concatenate(all_pred)
_,fscore_sklearn = f1_score(all_gold,all_pred,average=None)

import pdb;pdb.set_trace()
precision = true_pos / (true_pos + false_pos)
recall = true_pos / gold_pos
f_score = (2 * precision * recall) / (precision + recall)

print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'fscore: {f_score}')
