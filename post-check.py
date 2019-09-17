counter = 0

with open('all_acc.txt','r') as f:
    for line in f.readlines():
        wds,lbls = line.split('\t')
        wds = wds.split()
        lbls = lbls.split()
        if not len(wds)==len(lbls):
            print("mismatch")
            print(' '.join(wds),lbls)
            counter += 1
print('Mismatches: ',counter)