import numpy as np
import fem_tools as femt
import meshpy.triangle as triangle
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

####################
##    Generacion de la malla
###################

number_bound_points = 129 # numero de puntos en la frontera del circulo
radius = 1. # radio de circulo
center = [0.0,0.0] # coordenadas del centro del circulo
length=0.07

b_points,b_vertices=femt.CircleSegments(center,radius,number_bound_points)

# malla simple de un circulo con m puntos en la frontera

mesh, mesh_points, mesh_elements = femt.DoTriMesh(b_points,b_vertices,
                                               length, plot=None)

# Numeracion local en la frontera

b_mesh_points = mesh_points[:number_bound_points]

##############
##     Electrodos
##############

# Genera la numeracion de los electrodos e_l

num_electrodes =  ( number_bound_points  -1)/4

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

