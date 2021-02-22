import distmesh as dm
import numpy as np
fd = lambda p: np.sqrt((p**2).sum(1))-1.0
p, t = dm.distmesh2d(fd, dm.huniform, 0.2, (-1,-1,1,1), fig = 'gcf')
