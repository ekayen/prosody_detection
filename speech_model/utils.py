import torch
from torch.utils import data
import torch.nn.functional as F
import random
import numpy as np

random.seed(0)
torch.manual_seed(123)



class UttDataset(data.Dataset):
    def __init__(self, list_ids, utt_dict,label_dict,pad_len):
        self.list_ids = list_ids
        self.utt_dict = utt_dict
        self.label_dict = label_dict
        self.pad_len = pad_len

    def pad_left(self,arr):
        if arr.shape[0] < self.pad_len:
            dff = self.pad_len - arr.shape[0]
            arr = F.pad(arr,pad=(0,0,dff,0),mode='constant')
        else:
            arr = arr[arr.shape[0]-self.pad_len:]
        return arr

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        id = self.list_ids[index]
        X = self.pad_left(self.utt_dict[id])
        #X = self.utt_dict[id]
        y = self.label_dict[id]
        return X, y



"""
def test_pad_left(arr,pad_len):
    if arr.shape[0] < pad_len:
        dff = pad_len - arr.shape[0]
        arr = F.pad(arr,pad=(0,0,dff,0),mode='constant')
    else:
        arr = arr[arr.shape[0]-pad_len:]
    return arr


arr = np.arange(144).reshape(12,12)
arr = torch.tensor(arr)
arr1 = test_pad_left(arr,30)
arr2 = test_pad_left(arr,10)
"""
