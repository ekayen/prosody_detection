import os
from bs4 import BeautifulSoup
import subprocess
import pickle

def sec2hms(secs):
    mins = int(secs/60)
    s = secs - (mins*60)
    h = int(mins/60)
    m = mins - (h*60)
    return h,m,s
    
####################################
## BURNC                          ##
####################################


burnc_dir = '/group/corporapublic/bu_radio'
burnc_spkrs = ['f1a','f2b','f3a','m1b','m2b']
already_seen = set()
burnc_t = 0
for spkr in burnc_spkrs:
    for subdir, dirs, files in os.walk(os.path.join(burnc_dir,spkr)):
        for f in files:
            if f.endswith('sph') or f.endswith('spn'):
                if not f in already_seen:
                    tone_f = f.split('.')[0]+'.ton'
                    if os.path.exists(os.path.join(subdir,tone_f)):
                        secs = float(subprocess.check_output(['soxi', '-D',os.path.join(subdir, f)]).strip())
                        burnc_t += secs
                        already_seen.add(f)
                


print(f'burnc time in seconds: {burnc_t}')
h,m,s = sec2hms(burnc_t)
print(f'burnc time in h:mm:ss: {h}:{m}:{s}')


####################################
## SWBD INFOSTRUC                 ##
####################################

"""
swbd_t = 0
already_seen = set()
swbd_dir = '/group/corporapublic/switchboard/switchboard1/swb1'
swbd_list = 'stars/data/swbd/annotated_files.txt'
with open(swbd_list,'r') as f:
    for line in f.readlines():
        conv = line.split('.')[0][-4:]
        if not conv in already_seen:
            aud_file = os.path.join(swbd_dir,'sw0'+conv+'.sph')
            secs = float(subprocess.check_output(['soxi', '-D',aud_file]).strip())
            swbd_t += secs
            already_seen.add(conv)

print(f'infostruc swbd time in seconds: {swbd_t}')
h,m,s = sec2hms(swbd_t)
print(f'infostruc swbd time in h:mm:ss: {h}:{m}:{s}')
"""

####################################
## SWBD PROSODY                   ##
####################################


swbd_t = 0
already_seen = set()
swbd_dir = '/group/corporapublic/switchboard/switchboard1/swb1'
nxt_dir =  '/group/corporapublic/switchboard/nxt/xml'
convs = set()
#for fl in os.listdir(os.path.join(nxt_dir,'accent')):
#for fl in os.listdir(os.path.join(nxt_dir,'coreference')):
#for fl in os.listdir(os.path.join(nxt_dir,'markable')):
for fl in os.listdir(os.path.join(nxt_dir,'syntax')):
    convs.add(fl.strip())


   

def get_id(conv,idnum):
    return '_'.join([conv,idnum])

# Count audio duration        
for conv in convs:
    conv = conv.split('.')[0][-4:]
    if not conv in already_seen:
        aud_file = os.path.join(swbd_dir,'sw0'+conv+'.sph')
        try:
            secs = float(subprocess.check_output(['soxi', '-D',aud_file]).strip())
        except:
            print(f'not found {conv}')
        swbd_t += secs
        already_seen.add(conv)

print(f'prosody swbd time in seconds: {swbd_t}')
h,m,s = sec2hms(swbd_t)
print(f'prosody swbd time in h:mm:ss: {h}:{m}:{s}')

####################################
## SWBD COREF                     ##
####################################

swbd_t = 0
swbd_wds = 0
already_seen = set()
swbd_dir = '/group/corporapublic/switchboard/switchboard1/swb1'
swbd_list = 'annotated_files_coref.txt'
with open(swbd_list,'r') as f:
    for line in f.readlines():
        conv = line.split('.')[0][-4:]
        if not conv in already_seen:
            aud_file = os.path.join(swbd_dir,'sw0'+conv+'.sph')
            secs = float(subprocess.check_output(['soxi', '-D',aud_file]).strip())
            swbd_t += secs
            term_file_A = os.path.join(nxt_dir,'terminals',f'sw{conv}.A.terminals.xml')
            term_file_B = os.path.join(nxt_dir,'terminals',f'sw{conv}.B.terminals.xml')
            #print(subprocess.check_output(['grep', '-c','<word',term_file_A]).strip())
            wdsA = float(subprocess.check_output(['grep', '-c','<word',term_file_A]).strip())
            wdsB = float(subprocess.check_output(['grep', '-c','<word',term_file_B]).strip())
            swbd_wds = swbd_wds + wdsA + wdsB
            already_seen.add(conv)
        
            
print(f'coref swbd time in seconds: {swbd_t}')
h,m,s = sec2hms(swbd_t)
print(f'coref swbd time in h:mm:ss: {h}:{m}:{s}')
print(f'coref swbd words: {swbd_wds}')


####################################
## HCRC Maptask                   ##
####################################


hcrc_dir = '/group/corporapublic/maptask/hcrc'

hcrc_t = 0
for subdir, dirs, files in os.walk(hcrc_dir):
    for f in files:
        if f.endswith('.wav') and 'whole' in subdir:
            qdir = subdir.split('/')[-2]
            if qdir[0] == 'q' and qdir[-2] == 'c' and len(subdir.split('/'))==7:
                filepath = os.path.join(subdir,f)
                secs = float(subprocess.check_output(['soxi', '-D',aud_file]).strip())
                hcrc_t += secs
                #print(filepath)

hcrc_t = hcrc_t/2 # Because the stereo recordings are still full length, so you're double counting.
print(f'hcrc time in seconds: {hcrc_t}')
h,m,s = sec2hms(hcrc_t)
print(f'hcrc time in h:mm:ss: {h}:{m}:{s}')

            
