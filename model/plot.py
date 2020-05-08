import matplotlib.pyplot as plt
import numpy as np

# Bar plot:


# set width of bar
barWidth = 0.20

# set height of bar

full_cnn_lstm = [89.1,83.9,89.2,89.4,88.4,85.6,83.4]#,81.1]
full_cnn_only = [87.9,83.4,87.0,88.0,86.7,84.1,82.4]#,79.6]
threetok_cnn_lstm = [88.6,84.2,88.6,88.4,88.2,85.4,83.5]#,59.0]
threetok_cnn_only = [87.3,83.1,86.5,87.2,86.4,84.6,82.7]#,59.0]

buffer = 1
# Set position of bar on X axis
r1 = np.arange(len(full_cnn_lstm))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Make the plot
plt.bar(r1, full_cnn_lstm, color='darkblue', width=barWidth, edgecolor='white', label='Utterance-level context, CNN+LSTM')
plt.bar(r2, full_cnn_only, color='blue', width=barWidth, edgecolor='white', label='Utterance-level context, CNN only')
plt.bar(r3, threetok_cnn_lstm, color='darkred', width=barWidth, edgecolor='white', label='3-token context, CNN+LSTM')
plt.bar(r4, threetok_cnn_only, color='red', width=barWidth, edgecolor='white', label='3-token context, CNN only')
#plt.ylim(58,90)
plt.ylim(80,90)
# Add xticks on the middle of the group bars
plt.xlabel('', fontweight='bold')
#plt.xticks([r + barWidth for r in range(len(full_cnn_lstm))], ['All features','Intensity & voicing','Pitch & intensity','Pitch & voicing','Pitch','Intensity','Voicing','No features'],fontsize='x-large')
plt.xticks([r + barWidth for r in range(len(full_cnn_lstm))], ['All features','Intensity & voicing','Pitch & intensity','Pitch & voicing','Pitch','Intensity','Voicing'],fontsize='x-large')





# Create legend & Show graphic
plt.legend(fontsize='x-large')
plt.show()

##########################################################################

# Line plot for hparams:

filter_x = [5,7,9,11,13,15,17,19,21,23]
filter_y = [88.9,89.3,89.2,89.1,89.1,89.1,89.2,89.1,89.1,88.9]

layer_x = [2,3,4]
layer_y = [89.1,89.1,87.9]


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylim(87,90)
ax2 = ax1.twiny()

ax1.set_xlabel('CNN filter width',fontsize='x-large')
plt.ylabel('Accuracy')
plot1 = ax1.plot(filter_x,filter_y,'ro-',label='CNN filter width')
#plt.show()

ax1.set_xticks(filter_x)



#ax2.plot([2,3,4], np.ones(3)) # Create a dummy plot
ax2.cla()
ax2.set_xlabel(r"# CNN layers",fontsize='x-large')
ax2.set_xticks(layer_x)

#plt.xlabel('# CNN layers')
#plt.ylabel('Accuracy')
plot2 = ax2.plot(layer_x,layer_y,'bo-',label='# CNN layers')

plots = plot1+plot2
labs = [l.get_label() for l in plots]

ax1.legend(plots,labs,loc='lower left', shadow=True, fontsize='x-large')
#ax2.legend(loc='lower left', shadow=True, fontsize='x-large')

ax1.tick_params(axis='both', which='major', labelsize='x-large')
ax2.tick_params(axis='both', which='minor', labelsize='x-large')
ax1.grid(linestyle='--')

plt.show()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

#plt.rc('font', **font)

vocab_x = [5, 10, 25, 50, 100, 250, 500, 750, 1000]#, 1500, 2000]#, 2500, 3000]
performance_y = [70.5, 75.6, 79.4, 82.7, 84.6, 85.0, 84.9, 84.5, 84.5]#, 85.3, 85.2]#, 84.9, 85.1]

def epsilon_distort(x,epsilon):
    return (1/(1+epsilon-np.array(x)))

epsilon = 0.1
plt.xscale('log')
#plt.xticks(epsilon_distort(vocab_x,epsilon),labels=vocab_x)
plt.xticks(vocab_x,labels=vocab_x,fontsize='small')
plt.ylabel('Accuracy',fontsize='x-large')
plt.xlabel('Vocabulary size',fontsize='x-large')
#plt.plot(epsilon_distort(vocab_x,epsilon),performance_y,'bo-')
plt.plot(vocab_x,performance_y,'bo-')
plt.axhline(y=82.9,color='r', linestyle='dashed',label='Content word baseline')
plt.text(110,83.1,'Content word baseline',color='r',fontdict=font)
plt.grid(linestyle='--')
plt.show()
