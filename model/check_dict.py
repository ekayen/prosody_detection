import pickle

print('loading dict...')
swbd_dict = pickle.load(open('../data/swbd/swbd.unsplit.nps.pkl','rb'))
utt2nps = swbd_dict['utt2nps']
utt2old = swbd_dict['utt2old']
utt2new = swbd_dict['utt2new']
print('loaded.')

for utt in utt2old:
    try:
        assert len(utt2nps[utt]) == len(utt2old[utt])
        assert len(utt2nps[utt]) == len(utt2new[utt])
    except AssertionError:
        print(utt)
        print(utt2nps[utt])
        print(utt2old[utt])
        print(utt2new[utt])
        import pdb;pdb.set_trace()
print('All passed')
