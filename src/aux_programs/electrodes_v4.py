import numpy as np
import fem_tools as femt
import meshpy.triangle as triangle
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

####################
##    Generacion de la malla
###################

# Algoritmo para generar la malla

#radius = 10. # radio de circulo
#center = [0.0,0.0] # coordenadas del centro del circulo
#beta = 12 # numero de segmentaciones de un rayo
#radius_fraction = radius /beta
#pi =np.pi

#def PointsInCircum(r,n=12):
    #l = [(np.cos(2*pi/n*x)*r, np.sin(2*pi/n*x)*r) for x in range(n)]
    #l = np.array(l)
    #return l

#levels_list=[ 64, 64, 64, 64, 48, 48, 48, 32, 32, 32, 16, 12,1 ]
#number_bound_points = levels_list[0]

#cloud_points = [ ]

#for k in  range(len(levels_list)):
    #s = 12-k
    #index = levels_list[k]
    #cloud_points.extend( PointsInCircum(radius_fraction * s, index) )

#cloud_points = np.array(cloud_points )

#points = cloud_points
#tri = Delaunay(points)

#
#mesh_points = tri.points
#mesh_elements = tri.vertices


# salva las lista de puntos y vertices de la malla

#dataFile = open('mesh_points.txt', 'w')
#for eachitem in mesh_points:
    #dataFile.write( str(eachitem) + '\n')
#dataFile.close()

#dataFile = open('mesh_elements.txt', 'w')
#for eachitem in mesh_elements:
    #dataFile.write( str(eachitem) + '\n')
#dataFile.close()

#  Los datos de la malla se generaron con el algoritmo
# que esta en la parte superior

# Malla
#mesh_elements = np.loadtxt("meshelements.txt")
#mesh_points =  np.loadtxt("meshpoints.txt")

#print mesh_points

mesh_points = np.load('meshpoints_v1.npy')
mesh_elements = np.load('meshelements_v1.npy')


# Make a list of line segments:
# edge_points = [ ((x1_1, y1_1), (x2_1, y2_1)),
#                 ((x1_2, y1_2), (x2_2, y2_2)),
#                 ... ]
edge_points = []
edges = set()

def add_edge(i, j):
    """Add a line between the i-th and j-th points, if not in the list already"""
    if (i, j) in edges or (j, i) in edges:
        # already added
        return
    edges.add( (i, j) )
    edge_points.append(mesh_points[ [i, j] ])

# loop over triangles:
# ia, ib, ic = indices of corner points of the triangle
for ia, ib, ic in mesh_elements:
    add_edge(ia, ib)
    add_edge(ib, ic)
    add_edge(ic, ia)

edges = list(edges)

vertex = []
for k in edges:
    x= k[0]
    y =k[1]
    x= int(x)
    y = int(y)
    vector = [x,y]
    vertex.append(vector)


#mesh_elements = np.loadtxt("meshelements.txt")
#mesh_points = np.loadtxt("meshpoints.txt")

from meshpy.triangle import MeshInfo, build

mesh_info = MeshInfo()
mesh_info.set_points( mesh_points )
mesh_info.set_facets( vertex )
mesh = build(mesh_info)

mesh_points = np.array(mesh.points)
mesh_elements = np.array(mesh.elements)

# Genera la grafica de la malla
plt.gca().set_aspect('equal')
plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_elements,'go-')
plt.show()

##############
##     Electrodos
##############

## Genera la numeracion de los electrodos e_l

num_electrodes =  16
mesh_electrodes = [ ]

for k in range(num_electrodes):
    electrode_k = [ 4*k, 1 + 4*k,2+4*k  ]
    mesh_electrodes.append( electrode_k )

## Numeracion de los triangulos con dos vertices en electrodos

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

def triangles_of_electrode(index):
    """Da los vertices de los triangulos de la malla que inciden en el
            electodo numero index"""
    electrode = mesh_electrodes[index]
    N1, N2, N3 = electrode[0], electrode[1], electrode[2]
    P, Q = tindex(N1,N2), tindex(N2,N3)
    if P != Q:
        u = mesh_elements[P]
        v =mesh_elements[Q]
        t1  =[ mesh_points[u[0] ],  mesh_points[u[1] ],  mesh_points[u[2] ]    ]
        t2  =[ mesh_points[v[0] ],  mesh_points[v[1] ],  mesh_points[v[2] ]    ]
        t = [ t1, t2 ]
    else:
        u = mesh_elements[P]
        t1  =[ mesh_points[u[0] ],  mesh_points[u[1] ],  mesh_points[u[2] ]    ]
        t = [ t1 ]
    return t






