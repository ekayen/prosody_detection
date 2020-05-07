import os
import re
from string import punctuation
import pickle


burnc_path = '/group/corporapublic/bu_radio/'
discards = ('sil','endsil')
outpath = '../data/burnc/syl_dict.pkl'


def text_reg(word):
    remove = punctuation.replace('-','').replace('<','').replace('>','')
    word = word.lower().replace("'s","").replace("n't","").replace('/n','').replace('/v','')
    word = word.translate(str.maketrans('', '', remove))
    return word


def main():

    word_pattern = re.compile(r'>[a-zA-Z]+')
    syl_dict = {}
    for root, dirs, files in os.walk(burnc_path):
        for fil in files:
            if fil.endswith('.syl') and (fil.startswith('f') or fil.startswith('m')):
                print(fil)
                path = os.path.join(root,fil)
                with open(path) as f:
                    syl_count = 0
                    for line in f.readlines():
                        line = line.strip()
                        if line=='>':
                            syl_count += 1
                        elif word_pattern.match(line):
                            word = text_reg(line.lstrip('>'))
                            #word = line.lstrip('>')
                            if not word in discards:
                                if not word in syl_dict:
                                    syl_dict[word]=syl_count+1
                                    syl_count = 0
                                else:
                                    syl_count = 0
                            else:
                                syl_count = 0
                                    
    sorted_vals = sorted((value,key) for (key,value) in syl_dict.items())
    with open(outpath,'wb') as f:
        pickle.dump(syl_dict,f)

    print('Pickled syllable dictionary.')


if __name__=='__main__':
    main()
