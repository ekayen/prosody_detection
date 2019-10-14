import matplotlib.pyplot as plt
import numpy as np

turns = []

with open('../../data/all_acc.txt','r') as f:
    for line in f.readlines():
        txt,_ = line.split('\t')
        txt = txt.split()
        turns.append(len(txt))

utts = []

with open('../../data/utterances.txt','r') as f:
    for line in f.readlines():
        _,_,txt,_ = line.split('\t')
        txt = txt.split()
        utts.append(len(txt))

np_turns = np.array(turns)
np_utts = np.array(utts)
print('Number of turns:',len(turns))
print('Number of utterances:',len(utts))
print('Average length of turn:',np.average(np_turns))
print('Average length of utterance:',np.average(np_utts))
under_5_turns = (np_turns <= 3).sum()
under_5_utts = (np_utts <= 3).sum()
print('Number of turns with 5 words or fewer:',under_5_turns)
print('Percent of turns with 5 words or fewer:',under_5_turns/len(turns))

print('Number of utterances with 5 words or fewer:',under_5_utts)
print('Percent of utterances with 5 words or fewer:',under_5_utts/len(utts))

over_100_turns = (np_turns > 100).sum()
over_100_utts = (np_utts > 100).sum()
print('Number of turns over 100 words:',over_100_turns)
print('Percent of turns over 100 words:',over_100_turns/len(turns))
print('Number of utterances over 100 words:',over_100_utts)
print('Percent of utterances over 100 words:',over_100_utts/len(utts))

bns = 35
plt.hist(utts,range=[0,100],bins=bns,color='red')
plt.hist(turns,range=[0,100],bins=bns,color='blue')
plt.legend()


plt.show()
