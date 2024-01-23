import argparse
from model import SpeechEncoder
import torch
from evaluate import evaluate
import pickle
from utils import BurncDataset, load_vectors, set_seeds
import yaml
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--saved_model', help='path to saved model',default='results/CELoss_both_tenfold0_s256_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt')
    parser.add_argument("-c", "--config", help="path to config file", default='conf/mustc.yaml')
    parser.add_argument('-d', '--datasplit',
                        help='optional path to datasplit yaml file to override path specified in config')

    parser.add_argument('-hid', '--hidden_size',
                        help='hidden size for LSTM -- optional, overrides the one in the config')
    parser.add_argument('-f', '--frame_filter_size',
                        help='width of CNN filters -- optional, overrides the one in the config')
    parser.add_argument('-pad', '--frame_pad_size',
                        help='width of CNN padding -- optional, overrides the one in the config')
    parser.add_argument('-cnn', '--cnn_layers',
                        help='number of CNN layers -- optional, overrides the one in the config')
    parser.add_argument('-l', '--lstm_layers',
                        help='number of LSTM layers -- optional, overrides the one in the config')
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
    parser.add_argument('-s', '--stopword_baseline', action='store_true', default=False)
    parser.add_argument('-o', '--output_file', help='name of output file')
    parser.add_argument('-rs', '--seed', help='random seed -- optional, overrides the one in the config')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    if args.stopword_baseline:
        print('WARNING: STOPWORD BASELINE')

    cfg2arg = {'datasplit': args.datasplit,
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

    int_args = ['frame_filter_size', 'frame_pad_size', 'cnn_layers', 'lstm_layers', 'bottleneck_feats', 'hidden_size',
                'embedding_dim', 'vocab_size','seed']
    float_args = ['dropout', 'weight_decay', 'learning_rate']


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

    seed = cfg['seed']

    with open(cfg['all_data'], 'rb') as f:
        data_dict = pickle.load(f)


    #import pdb;pdb.set_trace()

    with open(cfg['datasplit'].replace('yaml', 'vocab'), 'rb') as f:
        vocab_dict = pickle.load(f)



    # Load text data:
    with open(cfg['datasplit'].replace('yaml', 'vocab'), 'rb') as f:
        vocab_dict = pickle.load(f)
    print(f'Original vocab size: {len(vocab_dict["w2i"])}')

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
                    print("WARNING: vocab size is not smaller than actual vocab")
        return w2i, i2w

    w2i, i2w = truncate_dicts(vocab_dict, cfg['vocab_size'])

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


    if 'overwrite_speech' in cfg:
        overwrite_speech = cfg['overwrite_speech']
    else:
        overwrite_speech = False

    if 'scramble_speech' in cfg:
        scramble_speech = cfg['scramble_speech']
    else:
        scramble_speech = False

    if 'stopwords_only' in cfg:
        stopwords_only = cfg['stopwords_only']
    else:
        stopwords_only = False

    if 'binary_vocab' in cfg:
        binary_vocab = cfg['binary_vocab']
    else:
        binary_vocab = False

    if 'ablate_feat' in cfg:
        ablate_feat = cfg['ablate_feat']
    else:
        ablate_feat = None

    model = SpeechEncoder(seq_len=cfg['frame_pad_len'],
                          batch_size=cfg['train_params']['batch_size'],
                          lstm_layers=cfg['lstm_layers'],
                          bidirectional=cfg['bidirectional'],
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



    model.load_state_dict(torch.load(args.saved_model))

    model.to(device)

    #testset = BurncDataset(cfg, data_dict, w2i, cfg['vocab_size'], mode='test',datasplit=datasplit,
    #                       overwrite_speech=overwrite_speech,stopwords_only=stopwords_only,binary_vocab=binary_vocab,
    #                       ablate_feat=ablate_feat)

    devset = BurncDataset(cfg, data_dict, w2i, cfg['vocab_size'], mode='dev',datasplit=datasplit,
                           overwrite_speech=overwrite_speech,stopwords_only=stopwords_only,binary_vocab=binary_vocab,
                           ablate_feat=ablate_feat)


    ### Stopword baseline here:
    with open(cfg['datasplit'].replace('yaml', 'stop'), 'rb') as f:
        stopword_list = pickle.load(f)

    acc = evaluate(cfg,devset, cfg['eval_params'], model, device,tok_level_pred=cfg['tok_level_pred'],noisy=True,
                   print_predictions=True,vocab_dict=vocab_dict,stopword_baseline=args.stopword_baseline,stopword_list=stopword_list,bootstrap_resample=True)

    
    datasplit_name = os.path.split(datasplit)[-1].split('.')[0]

    out_path = os.path.join(os.path.dirname(args.saved_model),f'{args.output_file}_{datasplit_name}.tsv')
    with open(out_path,'w') as f:
        f.write(f'\tepochs\ttrain_losses\ttrain_accs\tdev_accs\tdev_\n')
        f.write(f'0\t0\t0\t0\t{acc[0]}')
        print(f'wrote results to {out_path}')



if __name__=='__main__':
    main()
