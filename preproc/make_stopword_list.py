import sys
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import os
from string import punctuation

"""
with open('../data/burnc/burnc.pkl','rb') as f:
    burnc_dict = pickle.load(f)

import pdb;pdb.set_trace()
"""

def text_reg(word):
    remove = punctuation.replace('-', '').replace('<', '').replace('>', '')
    word = word.lower().replace("'s", "").replace("n't", "").replace('/n', '').replace('/v', '')
    word = word.translate(str.maketrans('', '', remove))
    if word in ('don','hasn','hadn','shouldn','couldn','wouldn','shan','weren',
                'didn','haven','isn','needn','aren','mustn','doesn','mightn','wasn','ain'):
        word = word.rstrip('n')

    return word

#vocab_path = '../data/burnc/splits'
#vocab_path = '../data/swbd/splits'
#vocab_path = '../data/swbd_acc/splits'
#vocab_path = '../data/swbd_new_only/splits'
vocab_path = '../data/swbd_kontrast/splits'

vocab_name = sys.argv[1]

pth = os.path.join(vocab_path,f'{vocab_name}.vocab')

with open(pth,'rb') as f:
    vocab_dict = pickle.load(f)

stopwords = set(stopwords.words('english'))
stopwords_idx = set()

for word in stopwords:
    word = text_reg(word)
    try:
        stopwords_idx.add(vocab_dict['w2i'][word])
    except:
        pass

stopwords_idx.add(vocab_dict['w2i']['PAD'])

out_pth = os.path.join(vocab_path,f'{vocab_name}.stop')
print(f'outpath = {out_pth}')
with open(out_pth,'wb') as f:
    pickle.dump(stopwords_idx,f)

