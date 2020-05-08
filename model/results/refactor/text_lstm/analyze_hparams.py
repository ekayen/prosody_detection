import pandas as pd
import os
import matplotlib.pyplot as plt

rootdir = '.'

acc_dict = {}
speed_dict = {}


emb_map = {'e100':100,
           'e300':200}

bottleneck_map = {'b10':10,
                  'b70':70,
                  'b100':100,
                  'b700':700,
                  'b1000':1000,
                  'b1500':1500,
                  'b2000':2000}

# Load all the results files and make a dict 
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith('.tsv'):
            path = os.path.join(subdir, file)
            df = pd.read_csv(path,sep='\t')
            max_acc=max(df['dev_accs'].tolist())
            max_row = df.loc[df['dev_accs'] == max_acc]
            epochs = max_row['epochs'].tolist()[0]
            acc_dict[file] = max_acc
            speed_dict[file] = epochs
top_accs = [(k, v) for k, v in sorted(acc_dict.items(), key=lambda item: item[1],reverse=True)][:5]
top_speeds = [(k, v) for k, v in sorted(speed_dict.items(), key=lambda item: item[1],reverse=False)][:5]

print(f'Top accuracy: {top_accs[0]}')
print(f'Speed: {speed_dict[top_accs[0][0]]}')
print('\n')
print(f'Other top accs: {top_accs[1:]}')


ACC_THRESHOLD = 0.6

# Check if there is a correlation between acc and cnn depth / filter width

bottleneck_acc = []
bottleneck_feats = []
emb_size = []
emb_acc = []
bottleneck_acc_dict = {}
emb_acc_dict = {}

for model in acc_dict:
    if acc_dict[model] >= ACC_THRESHOLD:
        for num in bottleneck_map:
            if num in model:
                b_num = bottleneck_map[num]
                bottleneck_feats.append(b_num)
                bottleneck_acc.append(acc_dict[model])
                if not b_num in bottleneck_acc_dict:
                    bottleneck_acc_dict[b_num] = [acc_dict[model]]
                else:
                    bottleneck_acc_dict[b_num].append(acc_dict[model])
            for num in emb_map:
                if num in model:
                    e_num = emb_map[num] 
                    emb_size.append(e_num)
                    emb_acc.append(acc_dict[model])
                    if not e_num in emb_acc_dict:
                        emb_acc_dict[e_num] = [acc_dict[model]]
                    else:
                        emb_acc_dict[e_num].append(acc_dict[model])


############################
## Plot cnn layers vs acc
############################


fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(bottleneck_feats,bottleneck_acc)
avg_size = []
avg_acc = []
for size in bottleneck_acc_dict:
    avg_size.append(size)
    avg_acc.append(sum(bottleneck_acc_dict[size])/len(bottleneck_acc_dict[size]))

plt.scatter(avg_size,avg_acc,color='red')
for xy in zip(avg_size, avg_acc):
    ax.annotate('(%s,%s)' % xy, xy=xy, textcoords='data') 
plt.show()

#####################################
## Plot filter size layers vs acc
#####################################
fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(emb_size,emb_acc)
avg_size = []
avg_acc = []
for sz  in emb_acc_dict:
    avg_size.append(sz)
    avg_acc.append(sum(emb_acc_dict[sz])/len(emb_acc_dict[sz]))

plt.scatter(avg_size,avg_acc,color='red')
for xy in zip(avg_size, avg_acc):
    ax.annotate('(%s,%s)' % xy, xy=xy, textcoords='data') 

import pdb;pdb.set_trace()
plt.show()

            
