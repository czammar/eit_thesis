# -*- coding: utf-8 -*-
""""
Genera una triangulacion  con cierto
numero de puntos en la frontera (Version 3)
Vease la seccion 4.2.2 de las notas

"""

import numpy as np
import fem_tools as femt # programa auxiliar
import meshpy.triangle as triangle
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

################################################################
##                 Malla - Version III
################################################################

number_bound_points = 64 # numero de puntos en la frontera del circulo
radius = 10. # radio de circulo
length=1.09 # longitud maxima por lado

#  Genera los vertices de un poligono regular de num_points lados 
#  inscrito en un circulo de radio radius y aristas unen a sus vertices
# usando el programa fem_tools.py

b_points, b_aristas=femt.CircleSegments([0.,0.], radius,number_bound_points)

## Genera la malla del circulo con number_bound_points en la frontera
## usando el programa fem_tools.py

mesh, mesh_points,mesh_elements = femt.DoTriMesh(b_points,b_aristas,
                                               length, plot=None)


################################################################
##         Grafica usando matplotlib.pyplot
################################################################

x, y ,triangles = mesh_points[:, 0], mesh_points[:, 1], mesh_elements

plt.figure()
plt.gca().set_aspect('equal')
plt.triplot( x, y, triangles, 'go-')
plt.title(' malla')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()

# Imprime el numero de nodos y elementos de la malla

print "Numero de nodos"
print len(mesh_points)
print "Numero de elementos"
print len(mesh_elements)

# salva las lista de puntos, elementos y aristas de la malla

np.save("meshpoints_v3",mesh_points)
np.save("meshelements_v3",mesh_elements)

# Este loop da las arista de la malla, para salvar esos datos  

edge_points = [] # lista vacia
edges = set() # establece un conjunto vacio

def add_edge(i, j):
    """Agrega una line entre el i-esimo y j-esimo puntos
        si no esta ya en la lista """
    if (i, j) in edges or (j, i) in edges:
        #  Si ya esta agregado en la lista no agrega nada
        return
    edges.add( (i, j) )
    edge_points.append(mesh_points[ [i, j] ])

# loop sobre los elementos de la malla:
# ia, ib, ic = indices de los nodos del triangulo
for ia, ib, ic in mesh_elements:
    add_edge(ia, ib)
    add_edge(ib, ic)
    add_edge(ic, ia)
edges = list(edges)

np.save("meshedges_v3",edges)

#############################################################

# NOTA: definiendo
#tol = 1.0e-10 # al ejecutar la secuencia
#
#for k in range(number_bound_points):
    #print la.norm(mesh_points[:number_bound_points][k] - b_points[k])
#
## arroja en todos los casos 0.0
##
## i.e. los  number_bound_points triangulos de la numeracion inducida en el
## programa por los puntos en la frontera es la misma que la de b_points
## Eso implica que los puntos en la frontera y su numeracion de vertices
## queda definida como

# Numeracion local en la frontera
# b_mesh_points = mesh_points[:number_bound_points]