# -*- coding: utf-8 -*-
""""
Genera una triangulacion simple con cierto
numero de puntos en la frontera

"""

import numpy as np
import jw_meshtools as mt
import meshpy.triangle as triangle
import numpy.linalg as la

number_bound_points = 32 # numero de puntos en la frontera del circulo
radius = 1. # radio de circulo
center = [0.0,0.0] # coordenadas del centro del circulo
length=0.2


## malla simple de un circulo con m puntos en la frontera
p,v=mt.CircleSegments( center,radius, number_bound_points)
mesh_points,mesh_elements = mt.DoTriMesh(p,v,length)

print len(mesh_elements)
print mesh_elements
#print len(mesh_elements)


####  Encuentra los triangulos de la frontera??
