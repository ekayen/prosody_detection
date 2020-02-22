import parselmouth
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import os

#datapath = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/burnc/'
datapath = '/afs/inf.ed.ac.uk/group/project/prosody/opensmile-2.3.0/'
soundfile = 'burnc/f1as05p1.wav'
#soundfile = 'burnc/f2bs18p8.wav'
#soundfile = 'burnc/m1brrlp2.wav'
featfile = 'tmp.csv'

snd = parselmouth.Sound(os.path.join(datapath,soundfile))
df = pd.read_csv(os.path.join(datapath,featfile),sep=';')

timestamps = df['frameTime'].tolist()
zcr = df['pcm_zcr_sma'].tolist()
voicing = df['voicingFinalUnclipped_sma'].tolist()
hnr = df['logHNR_sma'].tolist()
loudness = df['audspec_lengthL1norm_sma'].tolist()
rms = df['pcm_RMSenergy_sma'].tolist()
f0 = df['F0final_sma'].tolist()
#import pdb;pdb.set_trace()



 # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")


def draw_zcr(timestamps,zcr,color='blue'):
    plt.plot(timestamps, zcr, linewidth=5, color=color)
    plt.grid(False)
    plt.ylim(0)
    #plt.ylabel("zero-crossing rate")

spec = snd.to_spectrogram()
fig,ax = plt.subplots(1)
ax.set_yticklabels([])
ax.set_xticklabels([])
#draw_spectrogram(spec)
#plt.twinx()
#plt.title('Zero crossing rate')
draw_zcr(timestamps,zcr)
#draw_zcr(timestamps,hnr,'red')
#draw_zcr(timestamps,voicing,'green')
#draw_zcr(timestamps,loudness,'green')
#draw_zcr(timestamps,rms,'black')
#draw_zcr(timestamps,f0,'orange')
#plt.ylim(0,1.2)

plt.show()
