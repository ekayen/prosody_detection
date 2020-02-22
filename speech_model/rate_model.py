from model import SpeechEncoder
from utils import load_vectors,set_seeds,gen_model_name,BurncDatasetSyl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import argparse
import yaml
import pickle
import numpy as np
from torch.utils import data

torch.multiprocessing.set_sharing_strategy('file_system')

def gen_data(dataset,model,cfg,device,pool=False):
    print('Running dataset ...')
    train_params = cfg['train_params']

    datagen = data.DataLoader(dataset, **train_params)

    cnn_X = []
    lstm_X = []
    Y = []
    with torch.no_grad():

        for id, (speech, text, toktimes), labels, rate in datagen:

            speech, text, labels = speech.to(device), text.to(device), labels.to(device)

            curr_bat_size = speech.shape[0]

            hidden = model.init_hidden(curr_bat_size)
            output,_,post_cnn_feats,post_lstm_feats= model(speech, text, toktimes, hidden)



            post_cnn_feats = np.array(post_cnn_feats.detach().cpu())
            post_lstm_feats = np.array(post_lstm_feats.detach().permute(1,2,0).cpu())
            #import pdb;pdb.set_trace()

            if pool:
                #import pdb;pdb.set_trace()
                post_cnn_feats = np.amax(post_cnn_feats, axis=2) # pooling over time
                post_lstm_feats = np.amax(post_lstm_feats, axis=1) # pooling over hidden dim
                #import pdb;pdb.set_trace()

            post_cnn_feats = post_cnn_feats.reshape(curr_bat_size, -1)
            post_lstm_feats = post_lstm_feats.reshape(curr_bat_size, -1)

            #import pdb;pdb.set_trace()


            rate = np.array(rate.detach().cpu())
            cnn_X.append(post_cnn_feats)
            lstm_X.append(post_lstm_feats)
            Y.append(rate)

    print('done.')
    cnn_X = np.concatenate(cnn_X, axis=0)
    lstm_X = np.concatenate(lstm_X, axis=0)
    Y = np.concatenate(Y,axis=0)
    return cnn_X,lstm_X,Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--saved_model', help='saved model to load from disc')
    parser.add_argument("-c", "--config", help="path to config file", default='conf/cnn_lstm_pros.yaml')
    parser.add_argument('-d', '--datasplit',
                        help='optional path to datasplit yaml file to override path specified in config')
    parser.add_argument('-v', '--vocab_size',
                        help='vocab size -- optional, overrides the one in the config')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    cfg2arg = {'datasplit': args.datasplit,
               'vocab_size': args.vocab_size,
               'saved_model': args.saved_model,
               'config':args.config
               }

    int_args = ['vocab_size']

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
            else:
                cfg[arg] = cfg2arg[arg]



    with open(cfg['all_data'], 'rb') as f:
        data_dict = pickle.load(f)

    # report_hparams(cfg,datasplit)

    set_seeds(seed)

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
                if cfg['inputs'] == 'text' or cfg['inputs'] == 'both':
                    print("WARNING: vocab size is not smaller than actual vocab")
        return w2i, i2w

    w2i, i2w = truncate_dicts(vocab_dict, cfg['vocab_size'])

    if cfg['use_pretrained']:
        if cfg['embedding_dim'] == 100:
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

    model_name = gen_model_name(cfg, datasplit)
    print(f'Model: {model_name}')

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
                          cnn_layers=cfg['cnn_layers'],
                          inputs=cfg['inputs'],
                          embedding_dim=cfg['embedding_dim'],
                          vocab_size=cfg['vocab_size'],
                          bottleneck_feats=cfg['bottleneck_feats'],
                          use_pretrained=cfg['use_pretrained'],
                          weights_matrix=weights_matrix,
                          return_prefinal=True # Gives us access to features in the model right after the CNN and the LSTM
                          )

    print('Model built!')
    model.load_state_dict(torch.load(cfg['saved_model']))
    model.to(device)
    print('Pretrained model loaded!')

    trainset = BurncDatasetSyl(cfg, data_dict, w2i, cfg['vocab_size'], mode='train', datasplit=datasplit,
                               vocab_dict=vocab_dict)
    devset = BurncDatasetSyl(cfg, data_dict, w2i, cfg['vocab_size'], mode='dev', datasplit=datasplit,
                               vocab_dict=vocab_dict)

    #"""
    train_p_cnn_X, train_p_lstm_X, train_p_Y = gen_data(trainset, model, cfg, device, pool=True)
    dev_p_cnn_X, dev_p_lstm_X, dev_p_Y = gen_data(devset, model, cfg, device, pool=True)


    print('Fitting models...')
    cnn_reg_p = LinearRegression().fit(train_p_cnn_X, train_p_Y)
    lstm_reg_p = LinearRegression().fit(train_p_lstm_X, train_p_Y)
    print('done.')

    #import pdb;pdb.set_trace()

    print('Predicting ...')
    cnn_p_pred_Y = cnn_reg_p.predict(dev_p_cnn_X)
    lstm_p_pred_Y= lstm_reg_p.predict(dev_p_lstm_X)
    print('done.')

    print(f'R2 score for CNN w pooling: {r2_score(dev_p_Y, cnn_p_pred_Y)}')
    print(f'R2 score for LSTM w pooling: {r2_score(dev_p_Y, lstm_p_pred_Y)}')

    #"""

    train_cnn_X, train_lstm_X, train_Y = gen_data(trainset, model, cfg, device)
    dev_cnn_X, dev_lstm_X, dev_Y = gen_data(devset,model,cfg,device)


    datasets = {'train_cnn_X':train_cnn_X,
                'train_lstm_X':train_lstm_X,
                'train_Y':train_Y,
                'dev_cnn_X':dev_cnn_X,
                'dev_lstm_X':dev_lstm_X,
                'dev_Y':dev_Y}


    for dataset in datasets:
        with open(f'{dataset}.pkl','wb') as f:
            pickle.dump(datasets[dataset],f)

    print('Fitting models...')
    cnn_reg = LinearRegression().fit(train_cnn_X, train_Y)
    lstm_reg = LinearRegression().fit(train_lstm_X, train_Y)
    print('done.')

    #import pdb;pdb.set_trace()

    print('Predicting ...')
    cnn_pred_Y = cnn_reg.predict(dev_cnn_X)
    lstm_pred_Y= lstm_reg.predict(dev_lstm_X)
    print('done.')

    print(f'R2 score for CNN: {r2_score(dev_Y, cnn_pred_Y)}')
    print(f'R2 score for LSTM: {r2_score(dev_Y, lstm_pred_Y)}')



if __name__=="__main__":
    main()
