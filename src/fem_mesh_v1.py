""""
Genera una malla con cierta coleccion de puntos (Version 1)
Vease la seccion 4.1 de las notas


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay

################################################################
##         Malla - Version I
################################################################

radius = 10. # radio del circulo
center = np.array([[0.0,0.0]]) # centro del circulo
number_circles = 12 # no. de circulos internos que tendra la malla 
radius_fraction = radius / number_circles
pi =np.pi

def PointsInCircum(r,n):
    """Genera en la circunferencia de radio r las coordenadas
        los vertices de un poligono regular de n lados"""
    l = [(np.cos(2*pi/n*x)*r, np.sin(2*pi/n*x)*r) for x in range(n)]
    l = np.array(l)
    return l

# nodes_by_circle: tiene el no. de  nodos por circulo a considerar
# desde el mas grande hasta el de radio cero

nodes_by_circle = [ 64, 64, 64, 64, 48, 48, 48, 32, 32, 32, 16, 12, 1 ]
cloud_points = [ ]

# Loop que da un conjunto de puntos en el circulo de radio radius:
# Une los puntos de poligonos inscritos en circulos cuyos radios son
# multiplos de radius_fraction y cuyos numeros de vertices son 
# especificados en las entradas de nodes_by_circle

for k in  range(number_circles):
    s = number_circles - k
    index = nodes_by_circle[k]
    cloud_points.extend( PointsInCircum(radius_fraction * s, index) )

cloud_points.extend(center)

cloud_points = np.array(cloud_points )

 # Genera la triangulacion usando Delaunay y los puntos cloud_points
tri = Delaunay(cloud_points)

################################################################
##         Grafica usando Delaunay (vease numpytrian.py)
################################################################

# Hace una lista de los segmentos de linea entre los nodos de la malla
# edge_points = [ ((x1_1, y1_1), (x2_1, y2_1)),
#                 ((x1_2, y1_2), (x2_2, y2_2)), ... ]

edge_points = [] # lista vacia
edges = set() # establece un conjunto vacio

def add_edge(i, j):
    """Agrega una line entre el i-esimo y j-esimo puntos
        si no esta ya en la lista """
    if (i, j) in edges or (j, i) in edges:
        #  Si ya esta agregado en la lista, pasa 
        return
    edges.add( (i, j) )
    edge_points.append(cloud_points[ [i, j] ])

# loop sobre los elementos de la malla:
# ia, ib, ic = indices de los nodos del triangulo
for ia, ib, ic in tri.vertices:
    add_edge(ia, ib)
    add_edge(ib, ic)
    add_edge(ic, ia)

# Genera la grafica usando la herramienta LineCollection
lines = LineCollection(edge_points)

plt.figure()
plt.title('Malla Version I')
plt.gca().add_collection(lines)
plt.gca().set_aspect('equal')
plt.gca().set_xlim([-1-radius,radius+0.5])
plt.gca().set_ylim([-1-radius,radius+0.5])
plt.plot(cloud_points[:,0], cloud_points[:,1], 'o', hold=2)
plt.show()

# Imprime el numero de nodos y el numero de elementos de la malla

print "Numero de Nodos"
print len(tri.points)
print "Numero de Elementos"
print len(tri.vertices)

# salva las lista de puntos, elementos y aristas de la malla

np.save("meshpoints_v1",tri.points)
np.save("meshelements_v1",tri.vertices)
edges = list(edges)
np.save("meshedges_v1",edges)
