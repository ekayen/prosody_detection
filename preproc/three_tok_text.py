"""
Pull out three-token chunks of text and label them with the tone label of the center word.
"""
import os
from string import punctuation

burnc_dir = '/afs/inf.ed.ac.uk/group/project/prosody/burnc'


def text_reg(word):
    remove = punctuation.replace('-','').replace('<','').replace('>','')
    word = word.lower().replace("'s","").replace("n't","").replace('/n','').replace('/v','')
    word = word.translate(str.maketrans('', '', remove))
    return word

def load_word_file(wdfile):
    with open(wdfile, 'r',encoding="utf8", errors='ignore') as f:
        annotated_wds = f.read().split('#')[1]
        lines = [line.strip() for line in annotated_wds.split('\n') if not line == '']
        #print(lines)
        if lines:
            if len(lines[0].split())<3:
                lines = lines[1:]
            words = [text_reg(line.split()[2]) for line in lines]
            timestamps = [float(line.split()[0]) for line in lines]
            timestamps = [0] + timestamps
            #print("=============================")
            return words,lines,timestamps
        else:
            return None,None,None

def main():
    # Segment text into sentence-level utterances
    speakers = ['f1a','f1a','f2b','f3a','m1b','m2b','m3b','m4b']
    # Go through all the datafiles
    for sp in speakers:
        datadir = os.path.join(burnc_dir,sp)
        for subdir, dirs, files in os.walk(datadir):
            for file in files:
                # For each distinct paragraph, pull out the word file and the tone file
                if '.wrd' in file:
                    para_id = file.split('.')[0]
                    wordfile = os.path.join(subdir,file)
                    # Load tone file
                    tonefile = os.path.join(subdir,para_id+'.ton')
                    if os.path.exists(tonefile):
                        words,lines,timestamps = load_word_file(wordfile)




if __name__ == "__main__":
    main()

