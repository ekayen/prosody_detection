from utils import load_data
import numpy as np

def dummy_predict(seq):
    seq = seq.split()
    return np.random.randint(2,size=len(seq))

def evaluate(instance):
    seq, lbl = instance
    print(seq,lbl)
    prediction = dummy_predict(seq) # TODO eventually this'll probably need to be a np int represenatation of the seq
    print(prediction)
    #import pdb;pdb.set_trace()
    tp = np.sum(np.logical_and(prediction == 1, lbl == 1))
    fp = np.sum(np.logical_and(prediction == 1, lbl == 0))
    tn = np.sum(np.logical_and(prediction == 0, lbl == 0))
    fn = np.sum(np.logical_and(prediction == 0, lbl == 1))

    print(tp,tn,fp,fn)

def evaluate_all(filename):
    data = load_data(filename)
    evaluate(data[0])




def main():
    evaluate_all('data/all_acc.pickle')

if __name__=='__main__':
    main()