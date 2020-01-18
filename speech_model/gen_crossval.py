import os

#DATA_HOME = f'{SCRATCH_HOME}/stars/data'
DATA_HOME = f'../data/burnc/splits'
base_call = f"python3 train.py -c conf/cnn_lstm_best.yaml"

output_base = 'crossval'

datasplits = {'tenfold0':3000,
              'tenfold1':3000,
              'tenfold2':3000,
              'tenfold3':3000,
              'tenfold4':3000,
              'tenfold5':3000,
              'tenfold6':3000,
              'tenfold7':3000,
              'tenfold8':3000,
              'tenfold9':3000,
              'f2b_only0':1700,
              'f2b_only1':1700,
              'f2b_only2':1700,
              'f2b_only3':1700,
              'f2b_only4':1700,
              'f2b_only5':1700,
              'f2b_only6':1700,
              'f2b_only7':1700,
              'f2b_only8':1700,
              'f2b_only9':1700,
              'heldout_f1a':2600,
              'heldout_f2b':1600,
              'heldout_f3a':2800,
              'heldout_m1b':2700,
              'heldout_m2b':2700}

per_file = 8

calls = []
for datasplit in datasplits:
    call = (f'{base_call} '
            f' -d {os.path.join(DATA_HOME,datasplit)}.yaml'
            f' -v {datasplits[datasplit]}'
    )
    calls.append(call)

calls = [calls[i:i + per_file] for i in range(0, len(calls), per_file)]
    
for i,call in enumerate(calls):
    filename = f'{output_base}{i}.sh'
    print_call = ' &\n'.join(call)
    with open(filename,'w') as f:
        f.write(print_call)
