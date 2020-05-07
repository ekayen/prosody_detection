import pandas as pd
import yaml
import copy

df = pd.read_csv('configs.csv')
with open('../conf/burnc.yaml','r') as f:
    default_cfg = yaml.load(f,yaml.FullLoader)

for i,row in df.iterrows():
    cfg = copy.deepcopy(default_cfg)
    name = row['name']
    cfg['model_name'] = name
    vocab = row['vocab']
    cfg['vocab_size'] = vocab
    tok = row['tok']
    if tok=='text2labels':
        cfg['datafile'] = '../data/burnc/text2labels'
    elif tok=='text2labels_breath_tok':
        cfg['datafile'] = '../data/burnc/text2labels_breath_tok'
    elif tok=='text2labels_breath_sent':
        cfg['datafile'] = '../data/burnc/text2labels_breath_sent'
    emb = row['emb']
    if emb=='100':
        cfg['glove_path'] = '../data/emb/glove.6B.100d.txt'
        cfg['use_pretrained'] = True
    elif emb=='300':
        cfg['glove_path'] = '../data/emb/glove.6B.300d.txt'
        cfg['use_pretrained'] = True
    else:
        cfg['use_pretrained'] = False
    with open(name+'.yaml','w') as f:
        yaml.dump(cfg,f)

