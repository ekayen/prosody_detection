import pickle

dict_file = '../data/swbd/swbd.pkl'

label_dict = {1:0,
              0:0}

with open(dict_file,'rb') as f:
    swbd_dict = pickle.load(f)
    for utt in swbd_dict['utt2toks']:
        newness = swbd_dict['utt2new'][utt]
        num_ones = sum(newness)
        num_zeros = len(newness) - sum(newness)
        label_dict[0] = label_dict[0] + num_zeros
        label_dict[1] = label_dict[1] + num_ones

print(f'Num ones: {label_dict[1]}')
print(f'Num zeros: {label_dict[0]}')
if label_dict[0] > label_dict[1]:


import pdb;pdb.set_trace()