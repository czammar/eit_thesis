""""
Genera una triangulacion con cierto
numero de puntos en la frontera (Version 2)
Vease la seccion 4.2.1 de las notas

"""

import numpy as np
import meshpy.triangle as triangle
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay

################################################################
##                 Malla - Version II
################################################################

number_bound_points = 64 # no. de puntos en la frontera del circulo
radius = 10. # radio de circulo
origen = np.array([[0.,0.]]) # centro del circulo
beta = 10 # numero de segmentaciones de un rayo

# Genera puntos en la frontera del circulo de radio radius
def CirclePoints(center,radius,num_points=10):
  """ Genera los vertices de un poligono regular de num_points lados 
  inscrito en un circulo de radio radius"""
  t=np.linspace(0., 2.*np.pi, num_points, endpoint = False)
  # define points
  points=[(center[0]+radius*np.cos(angle),center[1]+
          radius*np.sin(angle)) for angle in t]
  return points

b_points = CirclePoints([0.,0.], radius, number_bound_points)

# lista auxiliar de indices
aux_indices = [ ( beta -l)*(1.0/beta )  for l in range(0,beta) ]

# Puntos puntos del poligono de number_bounds_points
# vertices, inscrito circulo de radio radius
points =  np.array(b_points)

# Lista de homotecias a los puntos de circulo de radio radius
indices = [ s* points  for s in  aux_indices ] 

# Lista los puntos generados por las homotecias, poniendo primero 
# los del circulo de radio radius
for i in range(0,beta-1):
    points = np.append( points,indices[i+1] , axis=0)

points = np.append( points,  origen , axis=0)

# Genera la triangulacion usando Delaunay y los puntos points 

tri = Delaunay(points)

################################################################
##         Grafica usando Delaunay (vease numpytrian.py)
################################################################

# Hace una lista de los segmentos de linea entre los nodos de la malla
# edge_points = [ ((x1_1, y1_1), (x2_1, y2_1)),
#                 ((x1_2, y1_2), (x2_2, y2_2)),.. ]

edge_points = [] # lista vacia
edges = set() # establece un conjunto vacio

def add_edge(i, j):
    """Agrega una line entre el i-esimo y j-esimo puntos
        si no esta ya en la lista """
    if (i, j) in edges or (j, i) in edges:
        #  Si ya esta agregado en la lista no agrega nada
        return
    edges.add( (i, j) )
    edge_points.append(points[ [i, j] ])

# loop sobre los elementos de la malla:
# ia, ib, ic = indices de los nodos del triangulo
for ia, ib, ic in tri.vertices:
    add_edge(ia, ib)
    add_edge(ib, ic)
    add_edge(ic, ia)

# Genera la grafica usando la herramienta LineCollection

lines = LineCollection(edge_points)
plt.figure()
plt.title('Malla Version II')
plt.gca().add_collection(lines)
plt.gca().set_aspect('equal')
plt.plot(points[:,0], points[:,1], 'o', hold=1)
plt.show()

# Imprime la cantidad de nodos y elementos de la malla
print "Numero de Nodos"
print len(tri.points)
print "Numero de Elementos"
print len(tri.vertices)

# salva las lista de puntos y vertices de la malla
# salva las lista de puntos, elementos y aristas de la malla

np.save("meshpoints_v2",tri.points)
np.save("meshelements_v2",tri.vertices)
edges = list(edges)
np.save("meshedges_v2",edges)

