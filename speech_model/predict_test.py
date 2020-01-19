import argparse
from model import SpeechEncoder
import torch
from evaluate import evaluate
import pickle
from utils import BurncDatasetSpeech
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", help="path to config file", default='conf/cnn_lstm_pros.yaml')
    parser.add_argument("-m" ,"--model", help="path to saved model", default='results/refactor/cnn_lstm/cnn_lstm_pros_gc_bi_tenfold0_s256_cnn3_lstm2_d3_f15_p7.pt')
    parser.add_argument("-d","--datasplit",help="path to datasplit file", default='../data/burnc/splits/tenfold0.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    seed = cfg['seed']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.datasplit:
        datasplit = args.datasplit
    else:
        datasplit = cfg['datasplit']

    with open(cfg['all_data'], 'rb') as f:
        data_dict = pickle.load(f)

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

    model.load_state_dict(torch.load(args.model))

    model.to(device)


    testset = BurncDatasetSpeech(cfg, data_dict, mode='test', datasplit=datasplit)
    print(datasplit)
    evaluate(testset, cfg['eval_params'], model, device,tok_level_pred=cfg['tok_level_pred'],noisy=True)



if __name__=='__main__':
    main()