import parselmouth
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import os
import pickle
import yaml
from utils import BurncDatasetSpeech
from torch.utils import data
import subprocess


feat = 1 # value from 0-5, representing one of the prosodic feats

data_file = '../data/burnc/burnc_utt.pkl'
config = 'conf/burnc_breath_open.yaml'

with open(data_file,'rb') as f:
    data_dict = pickle.load(f)
with open(config,'r') as f:
    cfg = yaml.load(f,yaml.FullLoader)

trainset = BurncDatasetSpeech(cfg, data_dict, mode='train')


datapath = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/burnc'

train_params = cfg['train_params']
traingen = data.DataLoader(trainset, **train_params)

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")


def draw_feat(timestamps,feat):
    plt.plot(timestamps, feat, linewidth=5, color='blue')
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("prosodic feature")


tmp_sound = 'tmp.wav'
for id, (batch, toktimes), labels in traingen:
    id = id[0] # TODO don't know why this is necessary
    para_id = id.split('-')[0]
    toktimes = [x for sublist in toktimes.tolist() for x in sublist]
    sentence = ' '.join([data_dict['tok2str'][tok_id] for utterance in batch for tok_id in data_dict['utt2toks'][id]])
    start = toktimes[0]
    end = toktimes[-1]
    soundfile = os.path.join(datapath,para_id+'.wav')
    subprocess.run(['sox',soundfile,tmp_sound,'trim',str(start),'='+str(end)])
    snd = parselmouth.Sound(tmp_sound)
    spec = snd.to_spectrogram()

    draw_spectrogram((spec))
    plt.twinx()

    feat_vec = [i for i in batch[:,:,feat:feat+1].squeeze().tolist()]
    timestamps = [i/100 for i in range(len(feat_vec))]
    draw_feat(timestamps,feat_vec)

    plt.show()


    subprocess.run(['rm',tmp_sound])


"""

df = pd.read_csv(os.path.join(datapath,featfile),sep=';')

timestamps = df['frameTime'].tolist()
#zcr = df['pcm_zcr_sma'].tolist()
#zcr = df['voicingFinalUnclipped_sma'].tolist()
#zcr = df['HarmonicsToNoiseRatioACFLogdB_sma'].tolist()
#zcr = df['logHNR_sma'].tolist()
zcr = df['audspec_lengthL1norm_sma'].tolist()
#import pdb;pdb.set_trace()


 # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")





draw_spectrogram(spec)
plt.twinx()

draw_zcr(timestamps,zcr)
plt.show()
"""