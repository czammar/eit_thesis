## -*- coding: utf-8 -*-
#"""
#Created on Thu Jan 23 17:28:54 2014

#@author: cesar zamora
#"""

import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt  

#---------------------------------------------------------------------------------------------
#                        Generacion de la malla -- Seccion 2.2
#---------------------------------------------------------------------------------------------

#-------------------------------------------------------------
#                 Parametros de la malla
#-------------------------------------------------------------

# Numero de nodos en la frontera (debe ser par)
nodes_boundary = 64 
half_nodes_boundary = nodes_boundary/2

# radio del circulo
radius = 5.

# centro del circulo
center = np.array([ [0.0,0.0] ]) 

# Numero de circulos internos que tendra la malla
number_circles = 13 

# Constante pi
pi  = np.pi

#-------------------------------------------------------------
#                   Puntos de la malla
#-------------------------------------------------------------

def PointsInCircum(r,n):
    """
    Genera n puntos sobre un circulo de radio r centrado en (0,0).
    
    Usando rotaciones, lista n puntos distribuidos uniformemente
    sobre el circulo, empezando desde (0,r) y continuando en el 
    sentido contrario a las manecillas del reloj.

    Parametros:
    r -- radio del circulo
    n -- numero de puntos 
    
    """
    # Angulo de rotacion
    theta = 2*pi/n
    
    #Genera los puntos variando el angulo en multiplos de theta
    l = [(np.cos(theta*x)*r, np.sin(theta*x)*r) for x in range(n)]
    l = np.array(l)
    return l

# Este loop une number_circles +1 conjuntos de n puntos 
# definidos sobre circulos de  radios cada vez mas chicos:
#        r, 
#       (number_circles )/(number_circles +1) *r,
#       (number_circles -1)/(number_circles +1) *r,
#       ,...,
#    ...1/(number_circles +1) *r
#
# Al final agrega el origen (0.0). La lista resultado, 
# mesh_points, constituye los nodos de la malla

#mesh_points = [ ]

#for i in range(number_circles+1):
    #multiple = (number_circles+1 - i )/(number_circles+1.) 
    #mesh_points.extend( PointsInCircum(multiple*radius, nodes_boundary) )
#mesh_points.extend(center)

mesh_points = PointsInCircum(5.,64)
mesh_points = np.array(mesh_points)

# Abscisas y ordenadas de los nodos
x= mesh_points[:,0]
y= mesh_points[:,1]

# Numero de nodos de la malla
number_nodes =  len(mesh_points)

edges = []

for i in range(number_nodes):
    if i != number_nodes-1:
        edges.append([  mesh_points[i], mesh_points[i+1] ]  )
    else:
        edges.append([  mesh_points[i], mesh_points[0] ]  )

edges = np.array(edges)

lines = LineCollection(edges)

plt.figure()
plt.gca().set_aspect('equal')
plt.gca().add_collection(lines)
plt.plot(x,y,'bo')
plt.title('Grafico de los electrodos')
#plt.xlabel('Eje X')
#plt.ylabel('Eje Y')
plt.gca().set_xlim([-1.-radius,radius+1.])
plt.gca().set_ylim([-1.-radius,radius+1.])
plt.show()
