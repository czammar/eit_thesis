# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:28:54 2014

@author: cesar zamora
"""

import numpy as np
from scipy.spatial import Delaunay

n = 100

points = np.zeros((n,2))

for i in range(0,n): #genera n puntos aleatorios dentro del cir de radio 1
    point = np.array([1.,1.])
    while np.linalg.norm(point) >= 1:
        g_point = 2 * np.random.random_sample((1, 2)) - 1
        point = g_point
    points[i] = point

#print points

#points = np.random.rand(n, 2) # 30 points in 2-d
tri = Delaunay(points)

# Make a list of line segments:
# edge_points = [ ((x1_1, y1_1), (x2_1, y2_1)),
#                 ((x1_2, y1_2), (x2_2, y2_2)),5
#                 ... ]
edge_points = []
edges = set()

def add_edge(i, j):
    """Add a line between the i-th and j-th points, if not in the list already"""
    if (i, j) in edges or (j, i) in edges:
        # already added
        return
    edges.add( (i, j) )
    edge_points.append(points[ [i, j] ])

# loop over triangles:
# ia, ib, ic = indices of corner points of the triangle
for ia, ib, ic in tri.vertices:
    add_edge(ia, ib)
    add_edge(ib, ic)
    add_edge(ic, ia)

# plot it: the LineCollection is just a (maybe) faster way to plot lots of
# lines at once
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

lines = LineCollection(edge_points)
plt.figure()
plt.title('Delaunay triangulation')
plt.gca().add_collection(lines)
plt.plot(points[:,0], points[:,1], 'o', hold=1)
#plt.xlim(-2, 2)
#plt.ylim(-2, 2)

# -- the same stuff for the convex hull

plt.show()
