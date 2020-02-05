import os
from math import floor

#DATA_HOME = f'{SCRATCH_HOME}/stars/data'
DATA_HOME = f'../data/burnc/splits'
base_call = f"python3 train.py -c conf/cnn_lstm_best.yaml"

output_base = 'filter'

filter_sizes = [5,7,9,11,13,15,17,19,21,23]

per_file = 8

calls = []
for filter_size in filter_sizes:
    pad_size = floor(filter_size/2)
    call = (f'{base_call} '
            f' -f {filter_size}'
            f' -pad {pad_size}'
    )
    calls.append(call)

calls = [calls[i:i + per_file] for i in range(0, len(calls), per_file)]
    
for i,call in enumerate(calls):
    filename = f'{output_base}{i}.sh'
    print_call = ' &\n'.join(call)
    with open(filename,'w') as f:
        f.write(print_call)
