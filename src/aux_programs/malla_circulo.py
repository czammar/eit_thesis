import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math
import distmesh as dm

def uniform_mesh_on_unit_circle():
    """Uniform Mesh on Unit Circle"""
    fd = lambda p: np.sqrt((p**2).sum(1))-1.0
    return dm.distmesh2d(fd, dm.huniform, 0.2, (-1,-1,1,1))

#plt.ion()
#pause = lambda : input('(press enter to continue)')
np.random.seed(1) # Always the same results

def fstats(p, t):
    print('%d nodes, %d elements, min quality %.2f'
        % (len(p), len(t), dm.simpqual(p,t).min()))

print('Uniform Mesh on Unit Circle')
p, t = uniform_mesh_on_unit_circle()
fstats(p,t)

x = p[:,0]*180/3.14159
y = p[:,1]*180/3.14159

plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(x, y, t, 'go-')
plt.title('triangulation usando T y P')
plt.xlabel('Longitud')
plt.ylabel('Latitude')

plt.show()


