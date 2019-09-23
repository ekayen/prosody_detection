import os
import string

datapath = 'data/mac_morpho'
NOUNS = ['N','NN','NPROP','PROPN']
out_txt = 'data/mac_morpho/all.txt'


data = []




for file in os.listdir(datapath):
    if '.txt' in file and not file == 'all.txt':
        fname = os.path.join(datapath, file)
        with open(fname,'r', encoding="latin-1") as f:
            lines = f.readlines()
            sent = []
            tags = []
            for line in lines:
                try:
                    txt, lbl = line.split("_")
                    if line.strip() == '._.' and not line.strip() == '':
                        data.append((sent, tags))
                        sent = []
                        tags = []
                    else:
                        sent.append(txt.strip())
                        if lbl.strip() in NOUNS:
                            tags.append(1)
                        else:
                            tags.append(0)

                        # import pdb;pdb.set_trace()
                except:
                    print("Omitting line: ",line)


with open(out_txt,'w') as f:
    for line in data:
        words = ' '.join(line[0])
        tags = [str(t) for t in line[1]]
        tags = ' '.join(tags)
        f.write(words+'\t'+tags+'\n')
#import pdb;pdb.set_trace()