from utils import UttDataset
from torch.utils import data
import torch
import numpy as np

def evaluate(dataset,dataloader_params,model,device):
    true_pos_pred = 0
    total_pred = 0
    dataloader = data.DataLoader(dataset, **dataloader_params)
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device),y.to(device)
            hidden = model.init_hidden(dataloader_params['batch_size'])
            output,_ = model(x,hidden)
            output = np.argmax(output.cpu())
            total_pred += 1
            if output.item() == y.item():
                true_pos_pred += 1
    acc = true_pos_pred/total_pred
    print('Accuracy: ',acc)
    return acc


