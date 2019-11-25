import os
import sys
import pandas as pd

target_dir = sys.argv[1]

max_accs = []
for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".tsv"):
             print(file)
             df = pd.read_csv(os.path.join(root, file),sep='\t')
             dev_accs = df['dev_accs'].tolist()
             max_accs.append(max(dev_accs))

print('Average of best performances:')
print(sum(max_accs)/len(max_accs))

