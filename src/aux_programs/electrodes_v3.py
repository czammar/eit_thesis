import numpy as np
import fem_tools as femt
import meshpy.triangle as triangle
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import Delaunay

####################
##    Generacion de la malla
###################


radius = 10. # radio de circulo
center = [0.0,0.0] # coordenadas del centro del circulo
beta = 12 # numero de segmentaciones de un rayo
radius_fraction = radius /beta
pi =np.pi

# Version 3

# Este algoritmo genera la nume de puntos

def PointsInCircum(r,n=12):
    l = [(np.cos(2*pi/n*x)*r, np.sin(2*pi/n*x)*r) for x in range(n)]
    l = np.array(l)
    return l

levels_list=[ 64, 64, 64, 64, 48, 48, 48, 32, 32, 32, 16, 12,1 ]
number_bound_points = levels_list[0]

cloud_points = [ ]

for k in  range(len(levels_list)):
    s = 12-k
    index = levels_list[k]
    cloud_points.extend( PointsInCircum(radius_fraction * s, index) )

cloud_points = np.array(cloud_points )

points = cloud_points

# Genera la triangulacion de la nube de puntos usando algoritmo Delaunay de Scipy
tri = Delaunay(points)
mesh_points = tri.points
mesh_elements = tri.vertices

# salva las lista de puntos y vertices de la malla

np.savetxt( "meshpoints.txt" ,mesh_points)
np.savetxt( "meshelements.txt" ,mesh_elements)

##############
##     Electrodos
##############

## Genera la numeracion de los electrodos e_l

num_electrodes =  16

mesh_electrodes = [ ]

for k in range(num_electrodes):
    electrode_k = [ 4*k, 1 + 4*k,2+4*k  ]
    mesh_electrodes.append( electrode_k )

# Numeracion de los triangulos con dos vertices en la frontera

def tindex(indexi, indexj):
    """Da el indice del primer triangulo que tiene a
     indexi y indexj como vertices"""
    for s in range( len(mesh_elements) ):
        listm =  list(mesh_elements[s])
        test_i, test_j = indexi in listm ,indexj in listm
        test_ij =  test_i and test_j
        if test_ij:
            index = s
            break
    return index

# Estos  loops listan  los triangulos con vertices en la frontera
#  evitando repetir indices en tal lista

mi_lista=[ ]

for i in range(number_bound_points):
    t= tindex(i,i+1)
    test_triangle =t not in mi_lista
    if test_triangle:
        mi_lista.append( t)

for s in range(len(mesh_elements)):
    t= tindex(0,number_bound_points-1)
    test_triangle =t not in mi_lista
    if test_triangle:
        mi_lista.append(t)

elements_electrodes_mesh= mi_lista

# Numeracion de los triangulos con dos vertices en electrodos

def tindex(indexi, indexj):
    """Da el indice del primer triangulo que tiene a
     indexi y indexj como vertices"""
    for s in range( len(mesh_elements) ):
        listm =  list(mesh_elements[s])
        test_i, test_j = indexi in listm ,indexj in listm
        test_ij =  test_i and test_j
        if test_ij:
            index = s
            break
    return index

# Estos  loops listan  los triangulos con vertices en electrodos
#  evitando repetir indices en tal lista

mi_lista=[ ]

for i in range(number_bound_points):
    t= tindex(i,i+1)
    test_triangle =t not in mi_lista
    if test_triangle:
        mi_lista.append( t)

for s in range(len(mesh_elements)):
    t= tindex(0,number_bound_points-1)
    test_triangle =t not in mi_lista
    if test_triangle:
        mi_lista.append(t)

elements_electrodes_mesh= mi_lista

