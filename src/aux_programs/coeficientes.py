""" Calcula los coeficientes de una funcion lineal a pedazos de la base
en un elemento triangular del malla Omega  """

import numpy as np
import fem_tools as femt
import meshpy.triangle as triangle
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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

# Genera la numeracion de los electrodos e_l

num_electrodes =  ( number_bound_points  -1)/4 -155

mesh_electrodes = [ ]

for k in range(num_electrodes):
    electrode_k = [  [0+4*k, 1 + 4*k ]  ,  [1+4*k,2+4*k ] ]
    mesh_electrodes.append( electrode_k )

#  Aproximacion de las integrales para FEM

def coef( index, number_triangle):
    """ Devuelve los coeficientes de la funcion lineal a pedazos
         phi_index en el triangulo T_{number_triangle}"""
    # Indices en numeracion global de los nodos del triangulo #
    indexes = mesh_elements[number_triangle]
    # Si  index es uno de los indices de la numeracion de los
    #nodos del triangulo, procede a calcular a,b,c
    #
    # En otro caso la funcion phi se anula en T_{number_triangule}
    if index in indexes:
        # Obtiene las coordenadas de los nodos del triangulo
        # Y calcula los coeficientes usando las formulas de las notas
        N1 = mesh_elements[indexes[0]]
        N2 = mesh_elements[indexes[1]]
        N3 = mesh_elements[indexes[2]]
        x1, x2, x3 = N1[0],N2[0],N3[0]
        y1, y2, y3 = N1[1],N2[1],N3[1]
        det = abs( (x1-x3)*(y2-y1)- (x1-x2)*(y3-y2) )
        invdet =  1./det
        # Calculo de los coeficientes dependiendo de
        # Que nodo corresponde al indice index
        #
        # Si es el index es el 1er nodo de T
        if index == indexes[0]:
            a = x2*y3-x3*y2
            b = y2-y3
            c = x3-x2
        # Si index es el 2do nodo de T
        elif index == indexes[1]:
            a = x3*y1-x1*y3
            b = y3-y1
            c = x1-x3
        # Si index es el 3er nodo de T
        else:
            a = x1*y2-x2*y1
            b = y1-y2
            c = x2-x1
        a, b, c = a*invdet, b*invdet, c*invdet
    else:
        a, b, c=0.,0.,0.
    return a, b, c


def sigma(x):
    """ Funcion impedancia electrica del tejido"""
    return x[0]*x[1]

a,b,c = coef(926,2009)

def sigmanablat(indexi,indexj, number_triangle):
    indexes = mesh_elements[number_triangle]
    test = ( indexi in indexes) and (indexj in indexes)
    if test:
        N1 = mesh_elements[indexes[0]]
        N2 = mesh_elements[indexes[1]]
        N3 = mesh_elements[indexes[2]]
        x1, x2, x3 = N1[0],N2[0],N3[0]
        y1, y2, y3 = N1[1],N2[1],N3[1]
        area_T= 0.5*abs( (x1-x3)*(y2-y1)- (x1-x2)*(y3-y2) )
        ai,bi,ci = coef(indexi,number_triangle)
        aj,bj,cj = coef(indexj,number_triangle)
        cen = (1./3.)*(N1+N2+N3)
        nabla_ij = bi*bj+ci*cj
        I = sigma(cen)*nabla_ij*area_T
    else:
        I=0.
    return

def sigmanabla(indexi,indexj):
    V_ij = [ ]
    for k in range(len(mesh_elements)):
        lugar_k =mesh_elements[k]
        test1 = indexi in lugar_k
        test2 = indexj in lugar_k
        test3 = test1 and test2
        if test3:
            V_ij.append(k)
    I=0.

#  Calculo de la integral en el dominio sumando las aportaciones en los elementos

    for number_triangle in range(len(V_ij)):
        I_k = sigmanablat(indexi,indexj, number_triangle)
        I = I+ I_k
    return I

