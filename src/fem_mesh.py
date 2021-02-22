# -*- coding: utf-8 -*-
""""
Genera una triangulacion simple con cierto
numero de puntos en la frontera

"""

import numpy as np
import fem_tools as femt
import meshpy.triangle as triangle
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

number_bound_points = 64 # numero de puntos en la frontera del circulo
radius = 10. # radio de circulo
center = [0.0,0.0] # coordenadas del centro del circulo
length=1.09

b_points,b_vertices=femt.CircleSegments(center,radius,number_bound_points)

## malla simple de un circulo con m puntos en la frontera

mesh, mesh_points,mesh_elements = femt.DoTriMesh(b_points,b_vertices,
                                               length, plot=None)

print "nodos"
print len(mesh_points)
print "elementos"
print len(mesh_elements)

####  Encuentra los triangulos de la frontera

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

##############################################################

# Numeracion local en la frontera
b_mesh_points = mesh_points[:number_bound_points]

# Numeracion de los triangulos con dos vertices en la frontera

mi_lista=[ ]

for i in range(number_bound_points):
    j,k=i,i+1
    for s in range(len(mesh_elements)):
        m_i = list(mesh_elements[s])
        j_i = m_i.count(j)
        k_i = m_i.count(k)
        if j_i == k_i and j_i== 1:
            mi_lista.append(s)

for s in range(len(mesh_elements)):
    m_i = list(mesh_elements[s])
    j_i = m_i.count(0)
    k_i = m_i.count(number_bound_points-1)
    if j_i == k_i and j_i== 1:
        mi_lista.append(s)

#Numeracion global de los triangulos con dos vertices en la frontera del circulo
b_mesh_elements = mi_lista


#############################################################
### Genera la grafica

### CASO 2 en 1
# nota: number_bound_points debe ser multiplo de 3

sl = number_bound_points/3
aux_lista = [ 3*k for k in range(sl)]
aux_lista_2 = [ b_mesh_elements[r] for r in aux_lista ]


### CASO 2 en 2 ###

# nota: number_bound_points debe ser multiplo de 4

#aux_lista = []
#aux_lista_complement = []

#k=0
#while k< number_bound_points:
    #aux_lista.append(k)
    #aux_lista.append(k+1)
    #aux_lista_complement.append(k+2)
    #aux_lista_complement.append(k+3)
    #k= k+4

aux_lista_2 = [ b_mesh_elements[r] for r in aux_lista ]

#####

x, y ,triangles = mesh_points[:, 0], mesh_points[:, 1], mesh_elements

z=0.9*np.ones(len(x))
for k in aux_lista:
    z[k]= 0.6

###  lista para la mascara borrando los triangulo de los electrodos
w=[]
for a in range(len(triangles)):
    w.append(None)
###
for k in aux_lista_2:
    w[k]= True

################ grafica a colores de la malla sin electrodos ##################

#plt.figure()
#plt.gca().set_aspect('equal')
#plt.tripcolor( x, y, triangles, z, mask=w, shading='faceted')
#plt.colorbar()
#plt.title('Coloracion de malla')
#plt.xlabel('Longitud')
#plt.ylabel('Latitud')
#plt.show()


########## malla sin colores

plt.figure()
plt.gca().set_aspect('equal')
plt.triplot( x, y, triangles, 'go-')
plt.title(' malla')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()