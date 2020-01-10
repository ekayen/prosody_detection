"""
By: Elizabeth Nielsen
"""
import pickle
from torch import nn
from torch.utils import data
import torch
import psutil
import os
from utils import *
from evaluate import evaluate,evaluate_lengths,baseline_with_len
import yaml
from model import SpeechEncoder
import numpy as np
import argparse
import pdb


def train(model,criterion,optimizer,trainset,devset,cfg,device,model_name):

    plot_data = {
        'time': [],
        'loss': [],
        'train_acc': [],
        'dev_acc': [],
    }
    recent_losses = []

    """
    print('Baseline eval....')
    plot_data['dev_acc'].append(evaluate(devset, cfg['eval_params'], model, device, tok_level_pred=cfg['tok_level_pred']))
    plot_data['train_acc'].append(evaluate(trainset, cfg['eval_params'], model, device, tok_level_pred=cfg['tok_level_pred']))
    plot_data['loss'].append(0)
    plot_data['time'].append(0)
    print('done')
    """

    #train_params = cfg['train_params']

    traingen = data.DataLoader(trainset, **cfg['train_params'])
    epoch_size = len(trainset)

    print('Training model ...')
    max_grad = float("-inf")
    min_grad = float("inf")
    for epoch in range(cfg['num_epochs']):
        timestep = 0
        tot_utts = 0
        tot_toks = 0
        for id, (batch, toktimes), labels in traingen:
            model.train()
            batch,labels = batch.to(device),labels.to(device)


            model.zero_grad()
            curr_bat_size = batch.shape[0]
            tot_utts += curr_bat_size

            hidden = model.init_hidden(curr_bat_size)
            output,_ = model(batch,toktimes,hidden)



            if cfg['tok_level_pred']:
                num_toks = [np.trim_zeros(np.array(toktimes[i:i + 1]).squeeze(), 'b').shape[0] - 1 for i in
                            range(toktimes.shape[0])] # list of len curr_bat_size, each element is len of that utterance
                seq_len = cfg['tok_pad_len']

                #import pdb;pdb.set_trace()

                # Flatten output and labels:

                output = output.squeeze().transpose(0,1) ####
                output = output.flatten()

                labels = labels.flatten()

                tmp_out = []
                tmp_lbl = []
                for i in range(curr_bat_size):
                    tmp_out.append(output[i * seq_len:i * seq_len + num_toks[i]])
                    tmp_lbl.append(labels[i * seq_len:i * seq_len + num_toks[i]])

                    # Experiment: don't flatten at all
                    #tmp_out.append(output[:num_toks[i],i].flatten())
                    #tmp_lbl.append(labels[i,:num_toks[i]].flatten())

                out = torch.cat(tmp_out)
                lbl = torch.cat(tmp_lbl)

                tot_toks += out.shape[0]

                #import pdb;pdb.set_trace()
                loss = criterion(out,lbl.float()) # TODO fix so that padding isn't included in loss calculation
            else:
                loss = criterion(output.view(curr_bat_size), labels.float())
            loss.backward()
            #plot_grad_flow(model.named_parameters())
            #plt.show()
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
                print('NaNs!!!!')
                import pdb;pdb.set_trace()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            recent_losses.append(loss.detach())

            if len(recent_losses) > 50:
                recent_losses = recent_losses[1:]

            timestep += curr_bat_size
            print_progress(timestep/epoch_size, info=f'Epoch {epoch}')

        # Print stuff!

        # time
        plot_data['time'].append(epoch)

        #loss
        train_loss = (sum(recent_losses)/len(recent_losses)).item()
        plot_data['loss'].append(train_loss)

        # train acc
        train_results = evaluate(trainset, cfg['eval_params'], model, device, tok_level_pred=cfg['tok_level_pred'],
                                 noisy=False)
        plot_data['train_acc'].append(train_results[0])

        # dev acc
        dev_results = evaluate(devset, cfg['eval_params'], model, device,tok_level_pred=cfg['tok_level_pred'],noisy=False)
        plot_data['dev_acc'].append(dev_results[0])

        print()
        print(f'Epoch: {epoch}\tTrain loss: {round(train_loss,5)}\tTrain acc: {round(train_results[0],5)}\tDev acc:{round(dev_results[0],5)}')
        #print(f'Total train utts: {train_results[3]}\ttrain toks: {train_results[1]},\ttotal correct: {train_results[2]}')
        #print(f'Total dev utts: {dev_results[3]}\tdev toks: {dev_results[1]},\ttotal correct: {dev_results[2]}')



    print('done')

    process = psutil.Process(os.getpid())
    print('Memory usage:',process.memory_info().rss/1000000000, 'GB')

    plot_results(plot_data,model_name,cfg['results_path'])

    print('Saving model ...')
    model_path = os.path.join(cfg['results_path'], model_name + '.pt')
    print(model_path)
    torch.save(model.state_dict(), model_path)
    print('done')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", help="path to config file", default='conf/cnn_lstm_pros.yaml')
    parser.add_argument('-d','--datasplit', help='optional path to datasplit yaml file to override path specified in config')

    parser.add_argument('-hid', '--hidden_size', help='hidden size for LSTM -- optional, overrides the one in the config')
    parser.add_argument('-f', '--frame_filter_size',
                        help='width of CNN filters -- optional, overrides the one in the config')
    parser.add_argument('-pad', '--frame_pad_size',
                        help='width of CNN padding -- optional, overrides the one in the config')
    parser.add_argument('-cnn', '--cnn_layers', help='number of CNN layers -- optional, overrides the one in the config')
    parser.add_argument('-l', '--lstm_layers', help='number of LSTM layers -- optional, overrides the one in the config')
    parser.add_argument('-dr', '--dropout', help='dropout -- optional, overrides the one in the config')
    parser.add_argument('-wd', '--weight_decay', help='weight decay -- optional, overrides the one in the config')
    parser.add_argument('-lr', '--learning_rate', help='learning rate -- optional, overrides the one in the config')
    parser.add_argument('-flat', '--flatten_method', help='method for flattening tokens -- optional, overrides the one in the config')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    cfg2arg = {'datasplit': args.datasplit,
               'frame_filter_size': args.frame_filter_size,
               'frame_pad_size': args.frame_pad_size,
               'cnn_layers': args.cnn_layers,
               'lstm_layers': args.lstm_layers,
               'dropout': args.dropout,
               'weight_decay': args.weight_decay,
               'learning_rate': args.learning_rate,
               'flatten_method': args.flatten_method,
               }

    int_args = ['frame_filter_size','frame_pad_size','cnn_layers','lstm_layers']
    float_args = ['dropout','weight_decay','learning_rate']

    seed = cfg['seed']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.datasplit:
        datasplit = args.datasplit
    else:
        datasplit = cfg['datasplit']

    for arg in cfg2arg:
        if cfg2arg[arg]:
            if arg in int_args:
                cfg[arg] = int(cfg2arg[arg])
            elif arg in float_args:
                cfg[arg] = float(cfg2arg[arg])
            else:
                cfg[arg] = cfg2arg[arg]

    model_name = gen_model_name(cfg,datasplit)
    print(f'Model: {model_name}')

    with open(cfg['all_data'], 'rb') as f:
        data_dict = pickle.load(f)

    #report_hparams(cfg,datasplit)

    set_seeds(seed)

    trainset = BurncDatasetSpeech(cfg, data_dict, mode='train', datasplit=datasplit)
    devset = BurncDatasetSpeech(cfg, data_dict, mode='dev',datasplit=datasplit)

    print('done')
    print('Building model ...')

    set_seeds(seed)

    model = SpeechEncoder(seq_len=cfg['frame_pad_len'],
                          batch_size=cfg['train_params']['batch_size'],
                          lstm_layers=cfg['lstm_layers'],
                          bidirectional=cfg['bidirectional'],
                          num_classes=1,
                          dropout=cfg['dropout'],
                          include_lstm=cfg['include_lstm'],
                          tok_level_pred=cfg['tok_level_pred'],
                          feat_dim=cfg['feat_dim'],
                          postlstm_context=cfg['postlstm_context'],
                          device=device,
                          tok_seq_len=cfg['tok_pad_len'],
                          flatten_method=cfg['flatten_method'],
                          frame_filter_size=cfg['frame_filter_size'],
                          frame_pad_size=cfg['frame_pad_size'],
                          cnn_layers=cfg['cnn_layers'])

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])

    set_seeds(seed)

    train(model, criterion, optimizer, trainset, devset, cfg, device, model_name)

    run_test = False

    if run_test:
        testset = BurncDatasetSpeech(cfg, data_dict, mode='test', datasplit=datasplit)
        evaluate(testset, cfg['eval_params'], model, device, tok_level_pred=cfg['tok_level_pred'], noisy=True)


if __name__=="__main__":
    main()