import pickle

swbd_file = '../data/swbd/swbd.pkl'

with open(swbd_file,'rb') as f:
    swbd = pickle.load(f)

utt2toks = swbd['utt2toks']
utt2frames = swbd['utt2frames']

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
