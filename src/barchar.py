##!/usr/bin/env python
## a bar plot with errorbars
#import numpy as np
#import matplotlib.pyplot as plt

#N = 5
#menMeans = (20, 35, 30, 35, 27)
#menStd =   (2, 3, 4, 1, 2)

#ind = np.arange(N)  # the x locations for the groups
#width = 0.35       # the width of the bars

#fig, ax = plt.subplots()
#rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

#womenMeans = (25, 32, 34, 20, 25)
#womenStd =   (3, 5, 2, 3, 3)
#rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)

## add some
#ax.set_ylabel('Scores')
#ax.set_title('Scores by group and gender')
#ax.set_xticks(ind+width)
#ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

#ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

#def autolabel(rects):
    ## attach some text labels
    #for rect in rects:
        #height = rect.get_height()
        #ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                #ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

#plt.show()


#########################################################

import numpy as np
import matplotlib.pyplot as plt

a = range(1,10)
b = range(4,13)
ind = np.arange(len(a))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(ind+0.25, b, 0.45, color='#7cfc00')

ax2 = ax.twinx()
ax2.bar(ind, a, 0.65, color='#b0c4de')

plt.xticks(ind, a)
ax.yaxis.set_ticks_position("right")
ax2.yaxis.set_ticks_position("left")

plt.tight_layout()
plt.show()

##############################################################

#import numpy as np
#import matplotlib.pyplot as plt

#fig = plt.figure()
#ax = fig.add_subplot(111)

### the data
#N = 5
#menMeans = [18, 35, 30, 35, 27]
#menStd =   [2, 3, 4, 1, 2]
#womenMeans = [25, 32, 34, 20, 25]
#womenStd =   [3, 5, 2, 3, 3]

### necessary variables
#ind = np.arange(N)                # the x locations for the groups
#width = 0.35                      # the width of the bars

### the bars
#rects1 = ax.bar(ind, menMeans, width,
                #color='black',
                ##yerr=menStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))

#rects2 = ax.bar(ind+width, womenMeans, width,
                    #color='red',
                    ##yerr=womenStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

## axes and labels
#ax.set_xlim(-width,len(ind)+width)
#ax.set_ylim(0,45)
#ax.set_ylabel('Scores')
#ax.set_title('Scores by group and gender')
#xTickMarks = [str(i) for i in range(1,6)]
#ax.set_xticks(ind+width)
#xtickNames = ax.set_xticklabels(xTickMarks)
#plt.setp(xtickNames, rotation=0, fontsize=10)

### add a legend
#ax.legend( (rects1[0], rects2[0]), ('Referencia', 'Obtenido') )

#plt.show()