import pandas as pd
import os
import matplotlib.pyplot as plt

rootdir = '.'

acc_dict = {}
speed_dict = {}



cnn_map = {'cnn1':1,
           'cnn2':2,
           'cnn3':3,
           'cnn4':4}

filter_map = {'f9':9,
              'f11':11,
              'f13':13,
              'f15':15,
              'f17':17,
              'f19':19,
              'f21':21,
              'f23':23}

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


ACC_THRESHOLD = 0.7

# Check if there is a correlation between acc and cnn depth / filter width

cnn_acc = []
cnn_depth = []
filter_size = []
filter_acc = []
cnn_acc_dict = {}
filter_acc_dict = {}
for model in acc_dict:
    if acc_dict[model] >= ACC_THRESHOLD:
        for num in cnn_map:
            if num in model:
                cnn_num = cnn_map[num]
                cnn_depth.append(cnn_num)
                cnn_acc.append(acc_dict[model])
                if not cnn_num in cnn_acc_dict:
                    cnn_acc_dict[cnn_num] = [acc_dict[model]]
                else:
                    cnn_acc_dict[cnn_num].append(acc_dict[model])
            for num in filter_map:
                if num in model:
                    f_num = filter_map[num] 
                    filter_size.append(f_num)
                    filter_acc.append(acc_dict[model])
                    if not f_num in filter_acc_dict:
                        filter_acc_dict[f_num] = [acc_dict[model]]
                    else:
                        filter_acc_dict[f_num].append(acc_dict[model])


import pdb;pdb.set_trace()
############################
## Plot cnn layers vs acc
############################


fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(cnn_depth,cnn_acc)
avg_depth = []
avg_acc = []
for depth in cnn_acc_dict:
    avg_depth.append(depth)
    avg_acc.append(sum(cnn_acc_dict[depth])/len(cnn_acc_dict[depth]))

plt.scatter(avg_depth,avg_acc,color='red')
for xy in zip(avg_depth, avg_acc):
    ax.annotate('(%s,%s)' % xy, xy=xy, textcoords='data') 
plt.show()

#####################################
## Plot filter size layers vs acc
#####################################
fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(filter_size,filter_acc)
avg_size = []
avg_acc = []
for sz  in filter_acc_dict:
    avg_size.append(sz)
    avg_acc.append(sum(filter_acc_dict[sz])/len(filter_acc_dict[sz]))

plt.scatter(avg_size,avg_acc,color='red')
for xy in zip(avg_size, avg_acc):
    ax.annotate('(%s,%s)' % xy, xy=xy, textcoords='data') 


plt.show()

            
