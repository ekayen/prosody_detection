import os
from bs4 import BeautifulSoup
import subprocess
import pickle
import numpy 

def sec2hms(secs):
    mins = int(secs/60)
    s = secs - (mins*60)
    h = int(mins/60)
    m = mins - (h*60)
    return h,m,s
    


swbd_t = 0
already_seen = set()
swbd_dir = '/group/corporapublic/switchboard/switchboard1/swb1'
nxt_dir =  '/group/corporapublic/switchboard/nxt/xml'
convs = set()
for fl in os.listdir(os.path.join(nxt_dir,'markable')):
    convs.add(fl.strip())

convs = list(convs)

# Check features of annotations
stat2nplen = {'old':[],
              'new':[],
              'med':[]}
stat2pos = {'old':[],
              'new':[],
              'med':[]}

ntid2pos = {}
ntid2pw = {}
term2pw = {}

def get_id(conv,idnum):
    return '_'.join([conv,idnum])

def extract_id_from_href(href):
    return href.split('(')[1].split(')')[0]
        

datadir = '/afs/inf.ed.ac.uk/group/project/prosody/prosody_detection/data/swbd'
dict_file = os.path.join(datadir,'swbd.unsplit.pkl')
swbd_dict = pickle.load(open(dict_file,'rb'))
tok2infostat = swbd_dict['tok2infostat']
utt2toks = swbd_dict['utt2toks']
utt2old = swbd_dict['utt2old']
utt2new = swbd_dict['utt2new']
utt2nps = {}

ntid2np = {}
ntid2nplen = {}
ntid2term = {}

for num,conv in enumerate(convs):
    #for num,conv in enumerate(convs[:5]):
    conv_id = conv.split('.')[0]

    # Get markables
    mark_contents = open(os.path.join(nxt_dir,'markable',conv),'r').read()
    mark_soup = BeautifulSoup(mark_contents, 'xml')
    marks = mark_soup.find_all('markable')

    # Get NPs
    syn_contents = open(os.path.join(nxt_dir,'syntax',conv.replace('markable','syntax')),'r').read()
    syn_soup = BeautifulSoup(syn_contents, 'xml')
    nps = [nt for nt in syn_soup.find_all('nt') if nt['cat']=='NP' or nt['cat']=='WHNP']
    
    # Map sentence ids to the NPs inside the sentences
    sentences = syn_soup.find_all('parse')
    for sent in sentences:
        sent_id = get_id(conv_id,sent['nite:id'])
        child_nps = [get_id(conv_id,nt['nite:id']) for nt in sent.find_all('nt') if nt['cat']=='NP']
        utt2nps[sent_id] = child_nps
    # Map NPs to their contents:
    curr_nps = []
    for np in nps:
        curr_nps.append(get_id(conv_id,np['nite:id']))
        ntid2np[get_id(conv_id,np['nite:id'])] = np
        ntid2nplen[get_id(conv_id,np['nite:id'])] = len(np.find_all('nite:child'))
        ntid2term[get_id(conv_id,np['nite:id'])] = [get_id(conv_id,child['href'].split('(')[1].split(')')[0]) for child in np.find_all('nite:child')] # nonterminal ID to terminals in that NT

    term_contents = open(os.path.join(nxt_dir,'terminals',conv.replace('markable','terminals')),'r').read()
    term_soup = BeautifulSoup(term_contents, 'xml')
    terms = term_soup.find_all('word')
    traces = [get_id(conv_id,tr['nite:id']) for tr in term_soup.find_all('trace')]
    sils = [get_id(conv_id,tr['nite:id']) for tr in term_soup.find_all('sil')]
    puncs = [get_id(conv_id,tr['nite:id']) for tr in term_soup.find_all('punc')]
    term2pos = dict(zip([get_id(conv_id,wd['nite:id']) for wd in terms],[wd['pos'] for wd in terms]))
    for terminal in terms:
        terminal_num = terminal['nite:id']
        start = terminal['nite:start']
        pos = terminal['pos']
        if not start=='non-aligned':
            term_id = get_id(conv_id, terminal_num)
            phonwords = terminal.find_all('nite:pointer')
            if phonwords:
                pw_num = extract_id_from_href(phonwords[0]['href'])
                pw_id = get_id(conv_id,pw_num)
                term2pw[term_id] = pw_id

    
    for idnum in curr_nps: #ntid2term:
        ntid2pos[idnum] = []
        ntid2pw[idnum] = []
        for t in ntid2term[idnum]:
            print(t)
            if t in term2pos:
                print('adding pos')
                ntid2pos[idnum].append(term2pos[t])
            elif t in traces:
                # print('adding trace')
                # ntid2pos[idnum].append('TRACE')
                pass
            elif t in sils:
                # print('adding sil')
                # ntid2pos[idnum].append('SIL')
                pass
            elif t in puncs:
                # print('adding punc')
                # ntid2pos[idnum].append('PUNC')
                pass
            else:
                print('adding nothing')
                # import pdb;pdb.set_trace()
            if t in term2pw:
                ntid2pw[idnum].append(term2pw[t])

    for mark in marks:
        marked_nts = [nt['href'].split('(')[1].split(')')[0] for nt in mark.find_all('nite:pointer')]
        marked_nt = get_id(conv_id,marked_nts[0])
        if mark.has_key('status'):
            stat = mark['status']
            children = mark.find_all('nite:pointer')
            if len(children)==1 and children[0]['href'].split('.')[2]=='terminals':
                pass  # This is the terminal being marked, not the NP, so don't count it.
            elif stat in stat2nplen:
                try:
                    stat2nplen[stat].append(ntid2nplen[marked_nt])
                    stat2pos[stat].extend(ntid2pos[marked_nt])
                except:
                    import pdb;pdb.set_trace()
        
        assert len(marked_nts)==1
    """
    if num==1:
        print('STOPPING EARLY FOR DEBUGGING')
        break
    """
    
utt2predold = {} # predictions using POS heuristic, where 1 = old
utt2prednew = {} # predictions using POS heuristic, where 1 = new

old_pos = set(['PRP'])
new_pos = set(['NN','NNS','NNP','NNPS'])
subset_conv_ids = set([conv.split('.')[0] for conv in convs[:1]])
for utt in utt2toks:
    print(utt)
    if True: #utt.split('_')[0] in subset_conv_ids:
        toks = utt2toks[utt]
        try:
            utt2predold[utt] = numpy.zeros((len(utt2toks[utt])))
            utt2prednew[utt] = numpy.zeros((len(utt2toks[utt])))
        except:
            import pdb;pdb.set_trace()
        print(len(list(utt2nps.keys())))
        try:
            assert utt in utt2nps
        except:
            import pdb;pdb.set_trace()
        for np in utt2nps[utt]:
            if ntid2pw[np]:
                start = toks.index(ntid2pw[np][0])
                end = toks.index(ntid2pw[np][-1]) + 1
                poses = ntid2pos[np]

                if old_pos.intersection(set(poses)):
                    utt2predold[utt][start:end] = 1
                elif new_pos.intersection(set(poses)):
                    utt2prednew[utt][start:end] = 1
        #import pdb;pdb.set_trace()

for stat in stat2nplen:
    print(stat)
    try:
        print(sum(stat2nplen[stat])/len(stat2nplen[stat]))
    except:
        import pdb;pdb.set_trace()
    poses = list(set(stat2pos[stat]))
    print(dict(zip(poses, [round(stat2pos[stat].count(pos)/len(stat2pos[stat]),4) for pos in poses])))


import pdb;pdb.set_trace()
            
