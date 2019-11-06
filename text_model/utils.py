import numpy as np
import pandas as pd
import pickle
import torch
import random
import operator
import string
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from seqeval.metrics import accuracy_score, classification_report,f1_score


def load_data(filename,seed=42,shuffle=True,debug=True,max_len=None):
    data = []
    if '.txt' in filename:
        with open(filename,'r') as f:
            for line in f.readlines():
                tokens,labels = line.split('\t')
                tokens = [tok.strip() for tok in tokens.split()]
                labels = [int(i) for i in labels.split()]
                if max_len:
                    tokens = tokens[:max_len]
                    labels = labels[:max_len]
                labels = np.array(labels,dtype=np.int32)
                data.append((tokens,labels))
    elif '.pickle' in filename:
        with open(filename,'rb') as f:
            data = pickle.load(f)
    else:
        print("File format not recognized.")
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    return data



def load_libri_data(filename,seed=42,shuffle=True,debug=True,max_len=None):
    HEADER = '<file>'
    NON_LABELED = '\tNA\t'
    FINAL_PUNC = '.?!'
    data = []
    lengths = []
    with open(filename,'r') as f:
        words = []
        labels = []
        for line in f.readlines():
            if not HEADER in line:
                word,label,_,_,_ = line.split('\t')
                if word in FINAL_PUNC:
                    lengths.append(len(words))
                    data.append((words,labels))
                    words = []
                    labels = []
                elif not NON_LABELED in line:
                    words.append(word.lower())
                    if label == '0':
                        labels.append(0)
                    else:
                        labels.append(1)
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    return data

def load_burnc_data(text2labels,seed=42,shuffle=True):

    df = pd.read_csv(text2labels, sep='\t', header=None)

    utt = df[0].tolist()
    text = df[1].tolist()
    labels = df[2].tolist()
    
    text =  [[tok for tok in line.split()] for line in text]
    labels = [[int(lbl) for lbl in line.split()] for line in labels]

    data = []
    for i in range(len(text)):
        data.append((text[i],labels[i]))
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    return data

def load_burnc_spans(spans,seed=42,shuffle=True):
    df = pd.read_csv(spans, sep='\t', header=None)

    spans = df[0].tolist()
    labels = df[1].tolist()

    text = [[tok for tok in line.split()] for line in spans]
    labels = [[int(line)] for line in labels]

    data = []
    for i in range(len(text)):
        data.append((text[i], labels[i]))
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    return data


def to_ints(data,vocab_size,wd_to_i=None,i_to_wd=None): # TODO add UNK and PAD (figure out what the pytorch defaults for these are, if any)

    num_wds = []
    num_lbls = []

    if not wd_to_i:
        wd_to_i = {'<PAD>': 1,
                   '<UNK>': 0}
        i_to_wd = {1: '<PAD>',
                   0: '<UNK>'}
        wd_counts = {}
        for example in data:
            wds,lbls = example

            for wd in wds:
                if not wd in wd_counts:
                    wd_counts[wd] = 1
                else:
                    wd_counts[wd] += 1
        wds_by_freq = sorted(wd_counts.items(), key=operator.itemgetter(1), reverse=True)
        print('Raw vocab: ',len(wds_by_freq))
        top_wds = [i[0] for i in wds_by_freq[:vocab_size]]
        counter = 2
        for wd in top_wds:
            wd_to_i[wd] = counter
            i_to_wd[counter] = wd
            counter += 1

    for example in data:
        wd_i = []
        wds,lbls = example
        for wd in wds:
            if wd in wd_to_i:
                wd_i.append(wd_to_i[wd])
            else:
                wd_i.append(wd_to_i['<UNK>'])
        num_wds.append(torch.tensor(wd_i,dtype=torch.long))
        num_lbls.append(torch.tensor(lbls,dtype=torch.long))

    return num_wds,num_lbls,wd_to_i,i_to_wd


class BatchWrapper:
    def __init__(self,in_iter,x,y):
        self.in_iter, self.x, self.y = in_iter,x,y

    def __iter__(self):
        for batch in self.in_iter:

            x = getattr(batch, self.x)
            y = getattr(batch, self.y)

            yield (x,y)

    def __len__(self):
        return len(self.in_iter)

def load_vectors(vector_file,wd_to_idx):
    vec_dict = {}
    with open(vector_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in wd_to_idx:
                wd_idx = wd_to_idx[word]
                vec_dict[wd_idx] = np.array(line[1:]).astype(np.float)
    return vec_dict

def shuffle_input_output(X,Y):
    data = list(zip(X,Y))
    random.seed(0)
    random.shuffle(data)
    X,Y = zip(*data)
    return(X,Y)

def make_batches(X,Y,batch_size,device):
    X,Y = shuffle_input_output(X,Y)
    start = 0
    end = batch_size
    batched_X = []
    batched_Y = []
    while end < len(X):
        X0 = X[start:end]
        Y0 = Y[start:end]
        X0 = pad_sequence(X0).to(device)
        Y0 = pad_sequence(Y0).to(device)
        batched_X.append(X0)
        batched_Y.append(Y0)
        start = end
        end = end + batch_size
    return(batched_X,batched_Y)

def make_single_labels(Y):
    batch_size = len(Y)
    out_labels = []
    for tens in Y:
        out_labels.append(tens[-1:])
    return torch.cat(out_labels).view(1,batch_size)

def make_nonseq_batches(X,Y,batch_size,device):
    X,Y = shuffle_input_output(X,Y)
    counter = 0
    start = 0
    end = batch_size
    batched_X = []
    batched_Y = []
    while end < len(X):
        X0 = X[start:end]
        Y0 = Y[start:end]
        X0 = pad_sequence(X0).to(device)
        Y0 = make_single_labels(Y0).to(device)
        batched_X.append(X0)
        batched_Y.append(Y0)
        start = end
        end = end + batch_size
    return(batched_X,batched_Y)

#def last_label(labels):


def majority_baseline(X,Y):
    preds = []
    for x in X:
        tmp = np.zeros(x.shape)
        #tmp = np.random.randint(2,size=x.shape[0])
        tmp.tolist()
        tmp = [str(int(i)) for i in tmp]
        preds.append(tmp)
    print("Majority baseline:")
    print('F1:',f1_score(Y, preds))
    print('Acc',accuracy_score(Y, preds))
    print(classification_report(Y, preds))


def plot_results(train_losses, train_accs, dev_accs, train_steps,model_name,results_dir=None):
    if not results_dir:
        results_dir = 'results'
    df = pd.DataFrame(dict(train_steps=train_steps,
                           train_losses=train_losses,
                           train_accs=train_accs,
                           dev_accs=dev_accs))

    with open("tmp.pkl", 'wb') as f:
        pickle.dump(df, f)

    ax = plt.gca()
    df.plot(kind='line', x='train_steps', y='train_losses', ax=ax)
    df.plot(kind='line', x='train_steps', y='train_accs', color='red', ax=ax)
    df.plot(kind='line', x='train_steps', y='dev_accs', color='green', ax=ax)

    plt.savefig(results_dir+'/{}.png'.format(model_name))
    plt.show()
    df.to_csv(results_dir+'/{}.tsv'.format(model_name), sep='\t')

def main():
    libri_file = '../data/libri/train_360.txt'
    ld,lengths = load_libri_data(libri_file)
    import matplotlib.pyplot as plt
    plt.hist(lengths,bins=50,range=(0,100))
    plt.show()
    import pdb;pdb.set_trace()

if __name__=='__main__':
    main()