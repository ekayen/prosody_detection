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


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            if self.y_vars is None:  # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)
