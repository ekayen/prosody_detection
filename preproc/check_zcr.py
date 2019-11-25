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
#soundfile = 'burnc/m1brrlp2.wav'
featfile = 'tmp.csv'

snd = parselmouth.Sound(os.path.join(datapath,soundfile))
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


def draw_zcr(timestamps,zcr):
    plt.plot(timestamps, zcr, linewidth=5, color='blue')
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("zero-crossing rate")

spec = snd.to_spectrogram()

draw_spectrogram(spec)
plt.twinx()

draw_zcr(timestamps,zcr)
plt.show()
