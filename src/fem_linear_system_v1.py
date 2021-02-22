import numpy as np
import meshpy.triangle as triangle

#####################################################################
##    Datos de la malla
#####################################################################

## Carga los archivos archivos meshpoints_v1.npy y meshvertex_v1.npy
## de la malla generada en fem_mesh_v1.py

#mesh_points = np.load('meshpoints_v1.npy')
#edges_float = np.load('meshedges_v1.npy')

## Loop para volver enteras las entradas de los vectores de edges_float
#ran_edges = range( len(edges_float) )
#edges  = [ map(int, edges_float[index]) for index in ran_edges]

## Genera usando los datos de los  la malla usando Meshpy

#mesh_info = triangle.MeshInfo()
#mesh_info.set_points( mesh_points )
#mesh_info.set_facets( edges )
#mesh = triangle.build(mesh_info, verbose=False)

#mesh_points = np.array(mesh.points)
#mesh_elements = np.array(mesh.elements)

#number_nodes = mesh_points.__len__() # len(mesh_points)
#number_elements = mesh_elements.__len__()# len(mesh_elements)



################################################################
##         Malla - Version I
################################################################
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay
radius = 10. # radio del circulo
center = np.array([[0.0,0.0]]) # centro del circulo
number_circles = 3 # no. de circulos internos que tendra la malla 
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

#nodes_by_circle = [ 64, 64, 64, 64, 48, 48, 48, 32, 32, 32, 16, 12, 1 ]
nodes_by_circle = [ 64, 64,64, 1 ] # la longitud debe ser number_circles + 1
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

print edge_points[0]
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




mesh_points = np.array(tri.points)
mesh_elements = np.array(tri.vertices)

number_nodes =  len(mesh_points)
number_elements =  len(mesh_elements)

#####################################################################
##     Grafica de la malla
#####################################################################
#import matplotlib.pyplot as plt
#plt.gca().set_aspect('equal')
#plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_elements,'go-')
#plt.show()

## NOTA:
## La malla fue definida de manera que  sus primeros 64 puntos
##  corresponden a  todos los puntos de la frontera de
## la region circular donde se resuelve el problema directo 
## Vease la seccion 4.1 de las notas 

#####################################################################
##     Electrodos
#####################################################################

## Genera la numeracion de los electrodos e_l

## NOTA: Por construccion los nodos de la frontera corresponden 
## en la num global del 0 al 63

## Los electrodos vamos a representarlos como triadas que agrupan 
## vertices en terminos de la numeracion global
## mesh_electrodes = [  [0,1,2], [4,5,6],...,[60,61,62] ]

num_electrodes =  16
mesh_electrodes = [ ]

for k in range(num_electrodes):
    electrode_k = [ 4*k, 1 + 4*k,2+4*k  ]
    mesh_electrodes.append( electrode_k )

##  Identifica los triangulos con dos vertices en electrodos
## y sus nodos para calculos posteriores

def tindex(indexi, indexj):
    """Da el indice del primer triangulo que tiene a
     indexi y indexj como vertices"""
# Busca en las triadas de la matriz de elementos el primer lugar donde
# aparece un triangulo con vertices indexi, indexj
    s=0
    while s < len(mesh_elements):
        listm =  list(mesh_elements[s])
        # localiza si indexi esta en el elemento de la lista vertices
        # si no actualiza, en caso afirmativo continua
        test_i = indexi in listm
        if test_i:
            # localiza si indexj esta en elemento de la lista
            # en caso afirmativo regresa el indice y rompe el loop
            # en caso negativo, actualiza
            test_j =indexj in listm
            if test_j:
                index = s
                break
            else:
                s= s+1
        else:
            s=s+1
    return index

def alltindex(indexi, indexj):
    """Da los indices de todos los triangulos que tienen a
     indexi y indexj como vertices"""
     # Busca en las triadas de la matriz de elementos los lugares donde
     # aparece un triangulo con vertices indexi, indexj
    indexes=[ ]
    s=0
    while s < len(mesh_elements):
        listm =  list(mesh_elements[s])
        # localiza si indexi esta en el elemento de la lista
        # si no actualiza, en caso afirmativo continua
        test_i = indexi in listm
        if test_i:
            # localiza si indexj esta en elemento de la lista de vertices
            # en caso afirmativo guarda el indice  en una lista
            # en caso negativo, actualiza
            test_j =indexj in listm
            if test_j:
                indexes.append(s)
                s=s+1
                indexes.append(s)
                s=s+1
            else:
                s= s+1
        else:
             s = s+1
    return indexes

def triangles_of_electrode(index):
    """Da los numeros de los triangulos de la malla que inciden en el
            electrodo numero index"""
    # Da los nodos del elemento cuyo indice es index
    electrode = mesh_electrodes[index]
    long_electrode = len(electrode)
    triangles_electrode= []
    ## Este loop obtiene los numeros de los triangulos que corresponden a 
    # los segmentos [N1,N2] \cup... \cup [Nk, N(k+)] que forman
    ## al electrodo de indice index, Ademas los guarda en una lista
    for i in range(long_electrode-1):
        r,s = i, i+1
        Nr,Ns= electrode[r], electrode[s]
        P = tindex(Nr,Ns)
        triangles_electrode.append(P)
    return triangles_electrode

#####################################################################
##         Vector de impedancias de contacto
#####################################################################

contact_impedance = 3.*np.ones((1,num_electrodes))


#####################################################################
##      Aproximacion de las integrales para FEM
#####################################################################

# Coeficientes de \phi_j en un triangulo T
# Vease la seccion 5.1 de las notas

def coef( index, number_triangle):
    """ Devuelve los coeficientes de la funcion lineal a pedazos
         phi_index en el triangulo T_{number_triangle}"""
    # Da los indices, respecto a la numeracion global,
    # de los nodos del triangulo cuyo numero es number_triangle
    indexes = mesh_elements[number_triangle]
    # Si  index es uno de los indices de la numeracion de los
    #nodos del triangulo, procede a calcular a,b,c
    #
    # En otro caso la funcion phi se anula en T_{number_triangule}
    #
    a, b, c=0.,0.,0.
    if index in indexes:
        # Obtiene las coordenadas de los nodos del triangulo
        # Y calcula los coeficientes usando las formulas de las notas
        N1 = mesh_points[indexes[0]]
        N2 = mesh_points[indexes[1]]
        N3 = mesh_points[indexes[2]]
        x1, x2, x3 = N1[0],N2[0],N3[0]
        y1, y2, y3 = N1[1],N2[1],N3[1]
        det = abs( (x1-x3)*(y2-y1)- (x1-x2)*(y3-y2) )
        invdet =  1.0/det
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
    return a, b, c

##  Funciones base \varphi_index en T_j

def phi(x, index, number_triangle):
    """ Define  la funcion phi_index en el triangulo
        cuyo indice es number_triangle"""
    a, b, c =coef(index, number_triangle)
    d = a+ b*x[0]  + c*x[1]
    return d



#  Funcion impedancia

def sigma(x):
    """ Funcion impedancia electrica del tejido"""
    # Aqui usamos esta para testear como ejemplo
    return 1.9


##  Aproximacion numerica de la integral
## \int_{\Omega}   \sigma ( \nabla \phi_i  * \nabla \phi_j ) dx,
## Vease la seccion 6.1 de las notas


def sigmanablat(indexi,indexj, number_triangle):
    """ Aproximacion numerica de la integral
     \int_{T_k}   \sigma ( \nabla \phi_i  * \nabla \phi_j ) dx,
    donde i = indexi, j = indexj, k = number_triangle,
    usando la regla de cuadratura de la notas """
    # Primero verifica que   \phi_i  ,\phi_j  tengan soporte en el triangulo
    # En caso de que no, la integral es cero
    indexes = mesh_elements[number_triangle]
    test = ( indexi in indexes) and (indexj in indexes)
    I=0.0
    #  Si  tienen soporte en el triangulo, aproxima la integral por
    # la regla de cuadratura
    if test:
        N1 = mesh_points[indexes[0]]
        N2 = mesh_points[indexes[1]]
        N3 = mesh_points[indexes[2]]
        x1, x2, x3 = N1[0],N2[0],N3[0]
        y1, y2, y3 = N1[1],N2[1],N3[1]
        area_T= 0.5*abs( (x1-x3)*(y2-y1)- (x1-x2)*(y3-y2) )
        ai,bi,ci = coef(indexi,number_triangle)
        aj,bj,cj = coef(indexj,number_triangle)
        centroid = (1./3.)*(N1+N2+N3)
        nabla_ij = bi*bj+ci*cj
        # Regla de cuadratura
        I = sigma(centroid)*nabla_ij*area_T
    return I

def sigmanabla(indexi,indexj):
    """ Aproximacion numerica de la integral
        \int_{\Omega}   \sigma ( \nabla \phi_i  * \nabla \phi_j ) dx,
        donde  i = indexi, j = indexj, usando la regla
        de cuadratura descrita en la notas"""
    # Obtenemos en la lista V_ij a todos los triangulos donde
    #  \phi_i  , \phi_j comparten soporte
    V_ij = alltindex(indexi, indexj)
    J=0.0
#  Calculamos  la integral en el dominio  Omega sumando las aportaciones
# en los elementos donde  \phi_i  , \phi_j comparten soporte
    for number_triangle in range( len(V_ij) ):
        J_k = sigmanablat(indexi,indexj, number_triangle)
        J = J+ J_k
    return J

## Aproximacion numerica de la integral
##  \int_{  e_l } \phi_i  * \phi_j dS
## Vease la seccion 6.2 de las notas


def  Dot_over_segment(indexi, indexj, number_triangle, x,y):
    """Calcula la integral \int_{ [x,y]} \phi_i  * \phi_j dS
    donde [x,y] es un segmento de electrodo,
    es decir, sobre el lado de un triangulo de la malla,
    usado la regla de cuadratura de las notas

    Indexi, indexj : numeros de las funciones  \phi_i, \phi_j' a integrar
    number_triangle:  numero del electrodo en que esta el segmento [x,y]
    x, y : punto inicial y final del segmento donde se integra,
    estos dos puntos tiene que ser nodos de la malla en la frontera
    """
    # Loop que prueba si los indices indexi e indexi conciden 
    #con alguno de x,y. Sino, la integral a aproximar es cero.
    # En caso afirmativo usa la regla de cuadratura aproximar
    #
    I= 0.
    # Obtiene los nodos de la malla que corresponden a los indices
    Ni, Nj = mesh_points[indexi], mesh_points[indexj] 
    nodes = np.array([x, y])
    # Prueba si los los nodos Ni,Nj corresponden a x o y
    test1, test2 = Ni in  nodes, Nj in nodes
    test = test1 and test2
    if test:
        # Define \phi_i  \phi_j  en el triagulo donde esta [x,y]
        def phi_i(z):
            return  phi(z, indexi, number_triangle)
        def phi_j(z):
            return  phi(z, indexj, number_triangle)
        # Regla de Cuadratura
        I1= phi_i(x)*phi_j(x)
        I2=phi_i(x)*( phi_j(y) -phi_j(x) ) + phi_j(x)*( phi_i(y) -phi_i(x) )
        I3 = ( phi_j(y) -phi_j(x) ) * ( phi_i(y) -phi_i(x) )
        I = I1+0.5*I2 + (1./3.)*I3
    return I

def  Dot_over_electrode(e_index, indexi, indexj):
    """Calcula la integral \int_{e_index} \phi_i* \phi_j  dS  """
    I = 0.0
    # Localiza los indices de los nodos de la malla 
    # que determinan al  electrodo e_index
    # e_index = [N1, N2] \cup  ... \..\cup.[ Nk, N(k+1)] 
    # 
    electrode = mesh_electrodes[e_index]
    # Este loop verifica si los indices  indexi, indexj forman
    # de los indices del electrodo e_index
    #
    # Sino, la integral es cero
    # En caso afirmativo, en cada segmento del electrodo integra
    if indexi in electrode:
        if indexj in electrode:
            long_electrode=  len(electrode)
            # Localiza los triangulos que corresponden a los segmentos
            # [Ni, N(i+1)] 
            t_electrode = triangles_of_electrode(e_index)
            #Este loop suma la aportacion del segmento [Ni,N(i+1)] 
            # para la integral sobre el electrodo
            for i in range(long_electrode-1):
                r,s = i, i+1
                #Obtiene lo nodos que corresponden al segmento [Ni,N(i+1)]
                Nr = mesh_points[ electrode[r]]
                Ns = mesh_points[ electrode[s]]
                # Calcula la integral sobre [Ni,N(i+1)]
                # Suma la aportacion de ambar para el electrodo
                I1= Dot_over_segment(indexi,indexj, t_electrode[i], Nr, Ns)
                I = I + I1
    return I


## Aproximacion numerica de la integral
##  \int_{  e_l } \phi_i    dS
## Vease la seccion 6.3 de las notas


def over_segment(indexi, number_triangule,x,y):
    """Calcula la integral \int_{ [x,y]} \phi_i   dS
    donde [x,y] es un segmento de electrodo,
    es decir, sobre el lado de un triangulo de la malla,
    usado la regla de cuadratura de las notas

    Indexi: indice de la funcion  \phi_i,  a integrar
    number_triangle:  numero del electrodo en que esta el segmento [x,y]
    x, y : punto inicial y final del segmento donde se integra,
    estos dos puntos tiene que ser nodos de la malla en la frontera
    """
    # Loop que prueba si el indice indexi concide
    #con alguno de x,y. Sino, la integral a aproximar es cero.
    # En caso afirmativo usa la regla de cuadratura aproximar
    I=0.0
    # Obtiene el nodo de la malla que corresponde al indice i
    Ni = mesh_points[indexi]
    # Prueba si el nodo Ni corresponden a x o y
    nodes = np.array([x, y])
    test =  Ni in nodes
    if test:
        def phi_i(xo):
            return  phi(xo, indexi, number_triangule)
        # Regla de Cuadratura
        I1= phi_i(x)
        I2= phi_i(y) - phi_i(x)
        I= I1+0.5*I2
    return I


def  over_electrode(e_index, indexi):
    """Calcula la integral \int_{e_index} \phi_idS  """
    I = 0.0
    # Localiza los indices de los nodos de la malla 
    # que determinan al  electrodo e_index
    # e_index = [N1, N2] \cup  ... \..\cup.[ Nk, N(k+1)] 
    # 
    electrode = mesh_electrodes[e_index]
    # Este loop verifica si los indices  indexi, indexj forman
    # de los indices del electrodo e_index
    #
    # Sino, la integral es cero
    # En caso afirmativo, en cada segmento del electrodo integra
    if indexi in electrode:
        long_electrode=  len(electrode)
        # Localiza los triangulos que corresponden a los segmentos
        # [Ni, N(i+1)] 
        t_electrode = triangles_of_electrode(e_index)
        #Este loop suma la aportacion del segmento [Ni,N(i+1)] 
        # para la integral sobre el electrodo
        for i in range(long_electrode-1):
            r,s = i, i+1
            #Obtiene lo nodos que corresponden al segmento [Ni,N(i+1)]
            Nr = mesh_points[ electrode[r]]
            Ns = mesh_points[ electrode[s]]
            # Calcula la integral sobre [Ni,N(i+1)]
            # Suma la aportacion de ambar para el electrodo
            J= over_segment(indexi, t_electrode[i], Nr, Ns)
            I = I + J
    return I


## Aproximacion numerica de la longitud del electrodo
##  |e_l|
## Vease la seccion 6.4 de las notas
## Nota asumimos que la longitud de todos los electrodos es uniforme


def long_electrode(index_e):
    "Aproxima la longitud de los electrodos"
    # Toma el arreglo de puntos que forman un electrodo index_e,
    # asumiendo que la longitud de estos es uniforme
    electrode = mesh_electrodes[index_e]
    # puntos que forman el electrodo
    p_electrode = np.array([mesh_points[i] for i in electrode])
    l = len(electrode)
    longitud =0.
    # Este loop suma las longitudes de las poligonales que forman
    # al electrodo elegido
    for j in range(l-1):
        longitud = longitud + np.linalg.norm(p_electrode[j]-p_electrode[j+1])
    return longitud

long_electrodes = long_electrode(0) 
# Notas: 
# 1) La longitud del lado de un poligono de n-lados inscritos en un
# circulo de radio r se calcula como l = sin(2 pi/n) * 2r
# 2) La longitud del electrodo es k*l, donde k es el numero de 
# lados que forman al electrodo

###############################################################################
### Matrices del sistema
###############################################################################

## Matriz M
def local_matrix_plus_centroid(number_triangle):
    """Esta funcion construye el centroide del triangulo 
    cuyo indice es number_triangle, da la matriz de gradientes
    \nabla \phi_i * \nabla \phi_j en el triangulo 
     multiplicada por el area del mismo y
    la funcion que va de la numeracion local a la
     global en tal triangulo"""
    # Triangulo con indice number_triangle
    K = mesh_elements[number_triangle]
    # Numeros de sus vertices 
    i,j,k = K[0],K[1],K[2]
    # Coordenadas de los vertices
    N1, N2,N3= mesh_points[i], mesh_points[j], mesh_points[k]
    # Centroide del triangulo
    centroid =  (1./3.)*(N1+N2+N3)
    # Abcisas y ordenadas de los vertices
    x1, x2, x3 = N1[0],N2[0],N3[0]
    y1, y2, y3 = N1[1],N2[1],N3[1]
    #area del triangulo
    area_T= 0.5*abs( (x1-x3)*(y2-y1)- (x1-x2)*(y3-y2) )
    # Coeficientes de las funciones base asociadas a los indices
    ai,bi,ci = coef(i,number_triangle)
    aj,bj,cj = coef(j,number_triangle)
    ak,bk,ck = coef(k,number_triangle)
    #Matriz de informacion local multiplicada por el area
    M_K = np.array([[ bi**2+ci**2,  bi*bj+ci*cj, bi*bk+ci*ck],
                    [ bj*bi+cj*ci,  bj**2+cj**2, bj*bk+cj*ck],
                    [ bk*bi+ck*ci,  bk*bj+ck*cj, bk**2+ck**2]])
    M_K = area_T * M_K
    def loc2glb(m):
        "Mapeo de numeracion local a global asociado al triangulo K"
        if m==0:
            return i
        elif m==1:
            return j
        else:
            return k
    return centroid, M_K, loc2glb

M = np.zeros((number_nodes,number_nodes))

# Este loop distribuye la aportacion local de las integrales 
# \int_{\T} \sigma (\nabla \phi_i * \nabla \phi_j) dx
# por triangulo para formar la matriz de masas cuyas entradas son 
#  \int_{\Omega} \sigma (\nabla \phi_i * \nabla \phi_j) dx

for number_triangle in range(len(mesh_elements)):
    centroid , M_K, loc2glb = local_matrix_plus_centroid(number_triangle)
    over_sigma = sigma(centroid)
    M_K = over_sigma* M_K
    for m in range(3):
        for n in range(3):
            M[loc2glb(m)][loc2glb(n)] = M[loc2glb(m)][loc2glb(n)] + M_K[m][n]


# Matriz Z
Z = np.zeros((number_nodes,number_nodes))
number_nodes_boundary =64
for i in range(number_nodes):
    for j in range(i,number_nodes):
        for e_index in range(num_electrodes):
             Z[i][j] = Z[i][j]+ (1./contact_impedance[0][e_index]) *Dot_over_electrode(e_index, i, j)
        Z[j][i] = Z[i][j]

# Matriz B
B = np.zeros((number_nodes,number_nodes))

# Este loop acopla las matrices de masas con las aportaciones en los
# electrodos
for i in range(number_nodes):
    for j in range(i,number_nodes):
        B[i][j] = M[i][j] + Z[i][j]
        B[j][i] = B[i][j]


# Matriz C 

C = np.zeros(( number_nodes, num_electrodes- 1))

for i in range(number_nodes):
    for j in range(num_electrodes-1):
        z= contact_impedance[0] 
        a = (1./z[0])* over_electrode(0,i)
        b = (1./z[j+1])* over_electrode(j+1, i)
        C[i][j] = - a + b 

# Matriz D^{-1}

inverse_D = np.zeros( (num_electrodes-1, num_electrodes-1) )

for k in range(num_electrodes-1):
    inverse_D[k][k ] = k+1.

# C^T D^{-1} C

CDC = np.dot( C ,np.dot(inverse_D,np.transpose(C) ))

####################################################
#######    Representacion grafica de las matrices
####################################################

from matplotlib.pyplot import figure, show, spy
import numpy

###  Plots independientes
#system_matrices = [B, M, Z,C, inverse_D, CDC, B + CDC]
system_matrices = [B, M, Z]
for matrix in system_matrices:
    figure()
    spy(matrix)
show()


## Todos en un mismo plot

#fig = figure()
#ax1 = fig.add_subplot(221)
#ax2 = fig.add_subplot(222)
#ax3 = fig.add_subplot(223)
#ax1.spy(B)
#ax2.spy(Z)
#ax3.spy(M)
#show()

print number_nodes