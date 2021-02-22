import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
fig = plt.figure(figsize=(5,5))
ax = plt.axes([0,0,1,1])
triangle1 = mpatches.Polygon(np.array([[0,1],[1,0],[1,1]]), fc="blue")
triangle2 = mpatches.Polygon(np.array([[-0.1,-1],[-2,-2],[-2,-1]]), fc="red")
ax.add_artist(triangle1)
ax.add_artist(triangle2)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plt.show()
