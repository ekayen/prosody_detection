import parselmouth
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import os
import pickle
import subprocess
import torch

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")


def draw_feat(timestamps,feat,color='blue'):
    plt.plot(timestamps, feat, linewidth=5, color=color)
    plt.grid(False)
    plt.ylim(0)
    #plt.ylabel("zero-crossing rate")

"""
BURNC checking:

#datapath = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/burnc/'
datapath = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/'
soundfile = 'burnc/f1as05p1.wav'
#soundfile = 'burnc/f2bs18p8.wav'
#soundfile = 'burnc/m1brrlp2.wav'
featfile = 'tmp.csv'

df = pd.read_csv(os.path.join(datapath,featfile),sep=';')

timestamps = df['frameTime'].tolist()
zcr = df['pcm_zcr_sma'].tolist()
voicing = df['voicingFinalUnclipped_sma'].tolist()
hnr = df['logHNR_sma'].tolist()
loudness = df['audspec_lengthL1norm_sma'].tolist()
rms = df['pcm_RMSenergy_sma'].tolist()
f0 = df['F0final_sma'].tolist()
#import pdb;pdb.set_trace()
"""




aud_dir = '/group/corporapublic/switchboard/switchboard1/swb1'
swbd_dict = '../../data/swbd_acc/swbd_acc.pkl'
with open(swbd_dict,'rb') as f:
    swbd = pickle.load(f)

for utt in swbd['utt2toks']:
    start = str(swbd['utt2toktimes'][utt][0])
    end = str(swbd['utt2toktimes'][utt][-1])
    conv = utt.split('_')[0][-4:]
    aud_file = os.path.join(aud_dir,'sw0'+conv+'.sph')

    subprocess.check_output(['sox',aud_file,'tmp.sph','trim',start,f'={end}'])
    snd = parselmouth.Sound('tmp.sph')
    #snd = parselmouth.Sound(os.path.join(datapath, soundfile))
    spec = snd.to_spectrogram()
    fig,ax = plt.subplots(1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    draw_spectrogram(spec)


    plt.twinx()
    plt.title('Zero crossing rate')
    feats = torch.cat([swbd['tok2pros'][tok] for tok in swbd['utt2toks'][utt]],dim=0)
    timestamps = range(0,feats.shape[0])
    timestamps = [i/100 for i in timestamps]
    zcr = list(feats[:,2:3])
    #import pdb;pdb.set_trace()
    draw_feat(timestamps,zcr)

    #draw_zcr(timestamps,hnr,'red')
    #draw_zcr(timestamps,voicing,'green')
    #draw_zcr(timestamps,loudness,'green')
    #draw_zcr(timestamps,rms,'black')
    #draw_zcr(timestamps,f0,'orange')
    #plt.ylim(0,1.2)
    fig.savefig(f'figs/{utt}_feat3.png')
    #import pdb;pdb.set_trace()

    plt.show()
