import matplotlib
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111)
rect1 = matplotlib.patches.Rectangle((0,0), 10, 10, color='yellow')
#rect2 = matplotlib.patches.Rectangle((0,150), 300, 20, color='red')
#rect3 = matplotlib.patches.Rectangle((-300,-50), 40, 200, color='#0099FF')
circle1 = matplotlib.patches.Circle((0,0), radius=10, color='#EB70AA')
ax.add_patch(rect1)
#ax.add_patch(rect2)
#ax.add_patch(rect3)
ax.add_patch(circle1)
plt.xlim([-11, 11])
plt.ylim([-11, 11])
plt.axis('scaled')

plt.gca().add_patch(circle0)
plt.gca().add_patch(circle1)

plt.show()