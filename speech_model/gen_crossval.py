import os

#DATA_HOME = f'{SCRATCH_HOME}/stars/data'
DATA_HOME = f'../data/burnc/splits'
base_call = f"python3 train.py -c conf/cnn_lstm_best.yaml"

output_base = 'crossval'

datasplits = ['tenfold0',
              'tenfold1',
              'tenfold2',
              'tenfold3',
              'tenfold4',
              'tenfold5',
              'tenfold6',
              'tenfold7',
              'tenfold8',
              'tenfold9',
              'f2b_only0',
              'f2b_only1',
              'f2b_only2',
              'f2b_only3',
              'f2b_only4',
              'heldout_f1a',
              'heldout_f2b',
              'heldout_f3a',
              'heldout_m1b',
              'heldout_m2b']

per_file = 6


calls = []
for datasplit in datasplits:
    call = (f'{base_call }'
            f'-d {datasplit}.yaml')
    calls.append(call)

calls = [calls[i:i + per_file] for i in range(0, len(calls), per_file)]
    
for i,call in enumerate(calls):
    filename = f'{output_base}{i}.sh'
    print_call = ' &\n'.join(call)
    with open(filename,'w') as f:
        f.write(print_call)
