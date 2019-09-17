import numpy as np
import pickle

def load_data(filename):
    data = []
    if '.txt' in filename:
        with open(filename,'r') as f:
            for line in f.readlines():
                tokens,labels = line.split('\t')
                tokens = [tok.strip() for tok in tokens.split()]
                labels = np.array([int(i) for i in labels.split()],dtype=np.int32)
                data.append((tokens,labels))
    elif '.pickle' in filename:
        with open(filename,'rb') as f:
            data = pickle.load(f)

    return data

