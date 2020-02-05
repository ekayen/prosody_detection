import matplotlib.pyplot as plt
import numpy as np

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
        'size'   : 22}

plt.rc('font', **font)

vocab_x = [5, 10, 25, 50, 100, 250, 500, 750, 1000]#, 1500, 2000]#, 2500, 3000]
performance_y = [70.5, 75.6, 79.4, 82.7, 84.6, 85.0, 84.9, 84.5, 84.5]#, 85.3, 85.2]#, 84.9, 85.1]

def epsilon_distort(x,epsilon):
    return (1/(1+epsilon-np.array(x)))

epsilon = 0.1
plt.xscale('log')
#plt.xticks(epsilon_distort(vocab_x,epsilon),labels=vocab_x)
plt.xticks(vocab_x,labels=vocab_x)
plt.ylabel('Accuracy',fontsize='x-large')
plt.xlabel('Vocabulary size',fontsize='large')
#plt.plot(epsilon_distort(vocab_x,epsilon),performance_y,'bo-')
plt.plot(vocab_x,performance_y,'bo-')
plt.axhline(y=82.4,color='r', linestyle='dashed',label='Stopword-only baseline')
plt.text(110,82.5,'Stopword-only baseline',color='r')
plt.grid(linestyle='--')
plt.show()
