"""
Anonymized
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
from torchsummary import summary
import time
from swbd_dataloader import SwbdSegDataset
from swbd_vocab_dict import create_vocab_dict
from model_seg import SpeechEncoderSeg
from evaluate_seg import evaluate_seg

print_time = False

def train(model,criterion,optimizer,trainset,devset,cfg,device,model_name,vocab_dict):


    plot_data = {
        'time': [],
        'loss': [],
        'train_acc': [],
        'dev_acc': [],
        'dev_prec': [],
        'dev_rec': [],
        'dev_f': []
    }

    recent_losses = []

    traingen = data.DataLoader(trainset, **cfg['train_params'])

    epoch_size = len(trainset)
    print(f'Epoch size: {epoch_size}')
    
    print('Training model ...')
    max_grad = float("-inf")
    min_grad = float("inf")
    best_dev_acc = 0
    for epoch in range(cfg['num_epochs']):
        t1 = time.time()
        timestep = 0
        tot_utts = 0
        tot_toks = 0

        for iden, frames, pause,dur, labels, toktimes, tokints in traingen:
            #toktimes = torch.transpose(toktimes,0,1)
            model.train()
            frames,pause,dur,tokints,labels = frames.to(device),pause.to(device),dur.to(device),tokints.to(device),labels.to(device) 

            model.zero_grad()
            curr_bat_size = frames.shape[0]
            tot_utts += curr_bat_size

            hidden = model.init_hidden(curr_bat_size)
            output,_ = model(frames,pause,dur,tokints,toktimes,hidden)

            if cfg['tok_level_pred']:
                num_toks = [np.trim_zeros(np.array(toktimes[i:i + 1]).squeeze(), 'b').shape[0] - 1 for i in
                            range(toktimes.shape[0])] # list of len curr_bat_size, each element is len of that utterance
                seq_len = cfg['tok_pad_len']

                #import pdb;pdb.set_trace()

                # Flatten output and labels:

                output = output.transpose(0,1)
                output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])

                labels = labels.flatten()

                tmp_out = []
                tmp_lbl = []
                for i in range(curr_bat_size):
                    output_seg = output[i * seq_len:i * seq_len + num_toks[i]]
                    tmp_out.append(output_seg)
                    tmp_lbl.append(labels[i * seq_len:i * seq_len + num_toks[i]])

                out = torch.cat(tmp_out)
                lbl = torch.cat(tmp_lbl)


                tot_toks += out.shape[0]

                loss = criterion(out, lbl)
            else:
                loss = criterion(output.view(output.shape[-2],output.shape[-1]), labels)#.float())
            loss.backward()

            if torch.sum(torch.isnan(frames)).item() > 0:
                print('NaNs!!!!')
                import pdb;pdb.set_trace()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            recent_losses.append(loss.detach())

            if len(recent_losses) > 50:
                recent_losses = recent_losses[1:]

            timestep += curr_bat_size
            print_progress(timestep/epoch_size, info=f'Epoch {epoch}')

        # time
        plot_data['time'].append(epoch)

        #loss
        train_loss = (sum(recent_losses)/len(recent_losses)).item()
        plot_data['loss'].append(train_loss)

        # train acc
        train_results = evaluate_seg(cfg, trainset, cfg['eval_params'], model, device, tok_level_pred=cfg['tok_level_pred'],
                                 noisy=False,print_predictions=True,vocab_dict=vocab_dict)
        plot_data['train_acc'].append(train_results[0])

        # dev acc

        dev_results = evaluate_seg(cfg, devset, cfg['eval_params'], model, device,tok_level_pred=cfg['tok_level_pred'],
                               noisy=False,print_predictions=True,vocab_dict=vocab_dict,prf=True)
        plot_data['dev_acc'].append(dev_results[0])
        plot_data['dev_prec'].append(dev_results[4])
        plot_data['dev_rec'].append(dev_results[5])
        plot_data['dev_f'].append(dev_results[6])

        print()
        print(f'Epoch: {epoch}\tTrain loss: {round(train_loss,5)}\tTrain acc: {round(train_results[0],5)}\tDev acc:{round(dev_results[0],5)}')
        t2 = time.time()
        if print_time: print(f'Epoch time: {t2-t1}')
        if dev_results[0] > best_dev_acc:
            print('Saving model ...')
            model_path = os.path.join(cfg['results_path'], f'{model_name}.pt')
            print(model_path)
            best_dev_acc = dev_results[0]
        torch.save(model.state_dict(), model_path)


    print('done')

    process = psutil.Process(os.getpid())
    print('Memory usage:',process.memory_info().rss/1000000000, 'GB')

    plot_results(plot_data,model_name,cfg['results_path'],p_r_scores=True)

    #print('Saving model ...')
    #model_path = os.path.join(cfg['results_path'], model_name + '.pt')
    #print(model_path)
    #torch.save(model.state_dict(), model_path)
    #print('done')

def main():

    mode = 'train'
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
    parser.add_argument('-flat', '--flatten_method',
                        help='method for flattening tokens -- optional, overrides the one in the config')
    parser.add_argument('-b', '--bottleneck_feats',
                        help='number of bottlneckfeats -- optional, overrides the one in the config')
    parser.add_argument('-e', '--embedding_dim',
                        help='number of bottlneckfeats -- optional, overrides the one in the config')
    parser.add_argument('-v', '--vocab_size',
                        help='vocab size -- optional, overrides the one in the config')
    parser.add_argument('-s', '--seed',
                        help='seed -- optional, overrides the one in the config')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    cfg2arg = {
               'frame_filter_size': args.frame_filter_size,
               'frame_pad_size': args.frame_pad_size,
               'cnn_layers': args.cnn_layers,
               'lstm_layers': args.lstm_layers,
               'dropout': args.dropout,
               'weight_decay': args.weight_decay,
               'learning_rate': args.learning_rate,
               'flatten_method': args.flatten_method,
               'bottleneck_feats': args.bottleneck_feats,
               'hidden_size': args.hidden_size,
               'embedding_dim': args.embedding_dim,
               'vocab_size': args.vocab_size,
               'seed': args.seed
               }

    int_args = ['frame_filter_size','frame_pad_size','cnn_layers','lstm_layers','bottleneck_feats','hidden_size','embedding_dim','vocab_size','seed']
    float_args = ['dropout','weight_decay','learning_rate']

    if args.seed:
        seed = int(args.seed)
    else:
        seed = cfg['seed']
    
    print(f'SEED: {seed}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    for arg in cfg2arg:
        if cfg2arg[arg]:
            if arg in int_args:
                cfg[arg] = int(cfg2arg[arg])
            elif arg in float_args:
                cfg[arg] = float(cfg2arg[arg])
            else:
                cfg[arg] = cfg2arg[arg]

    model_name = gen_model_name(cfg,seg=True)
    print(f'Model: {model_name}')

    set_seeds(seed)


    def truncate_dicts(vocab_dict, vocab_size):
        i2w = {}
        w2i = {}
        for i in range(vocab_size + 2):
            if i in vocab_dict['i2w']:
                w = vocab_dict['i2w'][i]
                i2w[i] = w
                w2i[w] = i
            else:
                if cfg['inputs']=='text' or cfg['inputs']=='both':
                    pass
                    #print("WARNING: vocab size is not smaller than actual vocab")
        return w2i, i2w

    
    w2i,i2w = create_vocab_dict(os.path.join(cfg['speech_dir'],'seg',f'turn_{mode}.txt'))
    vocab_dict = {'w2i':w2i,
                  'i2w':i2w}
    w2i,i2w = truncate_dicts(vocab_dict, cfg['vocab_size'])


    if cfg['use_pretrained']:
        print('using pretrained')
        if cfg['embedding_dim']==100:
            glove_path = cfg['glove_path_100']
        elif cfg['embedding_dim'] == 300:
            glove_path = cfg['glove_path_300']

        i2vec = load_vectors(glove_path, w2i)

        weights_matrix = np.zeros((cfg['vocab_size'] + 2, cfg['embedding_dim']))
        for i in i2w:
            try:
                weights_matrix[i] = i2vec[i]
            except:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(cfg['embedding_dim'],))
        weights_matrix = torch.tensor(weights_matrix)
    else:
        weights_matrix = None

    print('making trainset ...')
    trainset = SwbdSegDataset(cfg, w2i, mode='train')
    print('done')
    print('making devset ...')
    devset = SwbdSegDataset(cfg, w2i, mode='dev')
    print('done')

    print('Building model ...')

    set_seeds(seed)


    model = SpeechEncoderSeg(seq_len=cfg['frame_pad_len'],
                             batch_size=cfg['train_params']['batch_size'],
                             lstm_layers=cfg['lstm_layers'],
                             bidirectional=cfg['bidirectional'],
                             #num_classes=1, # TODO ! change to 2 for binary, n for n-ary
                             num_classes=cfg['num_classes'],
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
                             cnn_layers=cfg['cnn_layers'],
                             inputs=cfg['inputs'],
                             embedding_dim=cfg['embedding_dim'],
                             vocab_size=cfg['vocab_size'],
                             bottleneck_feats=cfg['bottleneck_feats'],
                             use_pretrained=cfg['use_pretrained'],
                             weights_matrix=weights_matrix)


    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])

    set_seeds(seed)
    
    train(model, criterion, optimizer, trainset, devset, cfg, device, model_name,vocab_dict)

    run_test = False

    if run_test:
        testset = BurncDatasetSpeech(cfg, data_dict, mode='test', datasplit=datasplit)
        evaluate_seg(cfg, testset, cfg['eval_params'], model, device, tok_level_pred=cfg['tok_level_pred'],
                 noisy=True, print_predictions=True)


if __name__=="__main__":
    main()
