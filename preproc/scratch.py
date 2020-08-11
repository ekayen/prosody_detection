import os
import re

mustc_inv = '/afs/inf.ed.ac.uk/group/project/prosody/mustc-investigation'

p = re.compile(r'[0-9]+\.[0-9]+')
abbrev = re.compile(f'[a-zA-Z]+\.[a-zA-Z]+\.')
bignum = re.compile(r'[0-9]+,[0-9]+')


for filename in os.listdir(mustc_inv):
    if filename.endswith('.en'):
        with open(os.path.join(mustc_inv,filename),'r') as f:
            linum = 0
            for line in f.readlines():
                wds = line.strip().split()
                for wd in wds:
                    if p.match(wd) or abbrev.match(wd) or bignum.match(wd):
                        print(wd)
                        print(linum)
                        print(filename)
                        import pdb;pdb.set_trace()
                linum += 1

