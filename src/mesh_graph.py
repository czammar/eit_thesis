from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy import sin, cos, pi
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0, 1, 1.0/51.0)
Y = np.arange(0, 1, 1.0/51.0)
X, Y = np.meshgrid(X, Y)
#Z = cos(pi*X) * sin(pi*Y) # solucion analitica de lines
Z  = Y/( (1.0+X)**2 + Y**2)# solucion analitica de laplace
ax.plot_surface(X, Y, Z,rstride=4, cstride=4, cmap='spectral')
plt.show()
