"""
Based very lightly on https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
by Sean Naren

Directly copied functions noted.

Modified by: Elizabeth Nielsen
"""
import pickle
from torch import nn
from torch.utils import data
import torch
import psutil
import os
from utils import UttDataset,plot_grad_flow,plot_results,UttDatasetWithId,UttDatasetWithToktimes,BurncDatasetSpeech,print_progress
from evaluate import evaluate,evaluate_lengths,baseline_with_len
import matplotlib.pyplot as plt
import random
import sys
import yaml
import math
import numpy as np
from model import SpeechEncoder


def train(model,criterion,optimizer,trainset,devset,cfg,device):
    plot_data = {
        'time': [],
        'loss': [],
        'train_acc': [],
        'dev_acc': [],
    }
    recent_losses = []


    print('Baseline eval....')
    plot_data['dev_acc'].append(evaluate(devset, cfg['eval_params'], model, device, tok_level_pred=cfg['tok_level_pred']))
    plot_data['train_acc'].append(evaluate(trainset, cfg['eval_params'], model, device, tok_level_pred=cfg['tok_level_pred']))
    plot_data['loss'].append(0)
    plot_data['time'].append(0)
    print('done')

    train_params = cfg['train_params']

    traingen = data.DataLoader(trainset, **train_params)
    epoch_size = len(trainset)

    print('Training model ...')
    max_grad = float("-inf")
    min_grad = float("inf")
    for epoch in range(cfg['num_epochs']):
        timestep = 0
        for id, (batch, toktimes), labels in traingen:

            model.train()
            batch,labels = batch.to(device),labels.to(device)

            model.zero_grad()
            curr_bat_size = batch.shape[0]
            hidden = model.init_hidden(curr_bat_size)
            output,_ = model(batch,toktimes,hidden)

            if cfg['tok_level_pred']:
                loss = criterion(output.view(output.shape[1],output.shape[0]), labels.float())
            else:
                loss = criterion(output.view(curr_bat_size), labels.float())
            loss.backward()
            #plot_grad_flow(model.named_parameters())

            """
            # Check for exploding gradients
            for n, p in model.named_parameters():
                if (p.requires_grad) and ("bias" not in n):
                    curr_min, curr_max = p.grad.min(), p.grad.max()
                    if curr_min <= min_grad:
                        min_grad = curr_min
                    if curr_max >= max_grad:
                        max_grad = curr_max
            print(f'min/max grad: {min_grad}, {max_grad}')
            """

            if torch.sum(torch.isnan(batch)).item() > 0:
                import pdb;pdb.set_trace()

            torch.nn.utils.clip_grad_norm(model.parameters(), 5)

            optimizer.step()
            recent_losses.append(loss.detach())

            if len(recent_losses) > 50:
                recent_losses = recent_losses[1:]

            timestep += 1
            print_progress(timestep/epoch_size, info=f'Epoch {epoch}')

        # Print stuff!

        # time
        plot_data['time'].append(epoch)

        #loss
        train_loss = (sum(recent_losses)/len(recent_losses)).item()
        plot_data['loss'].append(train_loss)

        # train acc
        train_acc = evaluate(trainset, cfg['eval_params'], model, device,tok_level_pred=cfg['tok_level_pred'],noisy=False)
        plot_data['train_acc'].append(train_acc)

        # dev acc
        dev_acc = evaluate(devset, cfg['eval_params'], model, device,tok_level_pred=cfg['tok_level_pred'],noisy=False)
        plot_data['dev_acc'].append(dev_acc)

        print()
        print(f'Epoch: {epoch}\tTrain loss: {round(train_loss,5)}\tTrain acc: {round(train_acc,5)}\tDev acc:{round(dev_acc,5)}')
        #if timestep % eval_every == 1 and not timestep==1:
        #evaluate(devset,eval_params,model,device,tok_level_pred=tok_level_pred)



    print('done')

    process = psutil.Process(os.getpid())
    print('Memory usage:',process.memory_info().rss/1000000000, 'GB')

    plot_results(plot_data,cfg['model_name'],cfg['results_path'])

def set_seeds(seed):
    print(f'setting seed to {seed}')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    if len(sys.argv) < 2:
        config = 'conf/burnc.yaml'
    else:
        config = sys.argv[1]

    with open(config, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)


    seed = cfg['seed']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(cfg['all_data'], 'rb') as f:
        data_dict = pickle.load(f)

    set_seeds(seed)

    trainset = BurncDatasetSpeech(cfg, data_dict, cfg['pad_len'], mode='train')
    devset = BurncDatasetSpeech(cfg, data_dict, cfg['pad_len'], mode='dev')

    print('done')
    print('Building model ...')

    set_seeds(seed)

    model = SpeechEncoder(seq_len=cfg['pad_len'],
                          batch_size=cfg['train_params']['batch_size'],
                          lstm_layers=cfg['lstm_layers'],
                          bidirectional=cfg['bidirectional'],
                          num_classes=1,
                          dropout=cfg['dropout'],
                          include_lstm=cfg['include_lstm'],
                          tok_level_pred=cfg['tok_level_pred'],
                          feat_dim=cfg['feat_dim'],
                          context=cfg['context'],
                          device=device)

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])

    set_seeds(seed)

    train(model, criterion, optimizer, trainset, devset, cfg, device)

if __name__=="__main__":
    main()