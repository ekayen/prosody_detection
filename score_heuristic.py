import pickle
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

ref_dict = pickle.load(open('data/swbd/swbd.unsplit.pkl','rb'))
pred_old_dict = pickle.load(open('utt2predold.pkl','rb'))
pred_new_dict = pickle.load(open('utt2prednew.pkl','rb'))

ref_new = []
ref_old = []
pred_new = []
pred_old = []

for utt in pred_old_dict:
    ref_new.extend(ref_dict['utt2new'][utt])
    pred_new.extend(pred_new_dict[utt].astype(np.int64).tolist())
    ref_old.extend(ref_dict['utt2old'][utt])
    pred_old.extend(pred_old_dict[utt].astype(np.int64).tolist())

new_results = precision_recall_fscore_support(pred_new,ref_new)
old_results = precision_recall_fscore_support(pred_old,ref_old)
print('new')
print(new_results)

print('old')
print(old_results)
import pdb;pdb.set_trace()
