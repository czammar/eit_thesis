import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

nb = 64 # Numero de nodos en la frontera (debe ser par)
number_nodes_boundary =nb
aux_nb = (nb/2)
radius = 10. # radio del circulo
center = np.array([[0.0,0.0]]) # centro del circulo
number_circles = 2 # no. de circulos internos que tendra la malla
radius_fraction = radius / number_circles
pi =np.pi

#------------------------------------------------------------------------------
#                            Puntos de la malla
#------------------------------------------------------------------------------
def PointsInCircum(r,n):
    """Genera en la circunferencia de radio r las coordenadas
        los vertices de un poligono regular de n lados"""
    l = [(np.cos(2*pi/n*x)*r, np.sin(2*pi/n*x)*r) for x in range(n)]
    l = np.array(l)
    return l

# Loop encargado de reunir los puntos de los niveles, agregando el origen
# al final; x, y son las listas de abcisas y ordenadas de los puntos
# para efectos de la grafica de la malla
cloud_points = [ ]
for i in range(number_circles+1):
    r= 1- (i /(number_circles+1.) )
    cloud_points.extend( PointsInCircum(radius*r, nb) )
cloud_points.extend(center)

cloud_points = np.array(cloud_points )
x= cloud_points[:,0]
y= cloud_points[:,1]

#------------------------------------------------------------------------------
#                            Triangulos de la malla
#------------------------------------------------------------------------------


def triangles_level(level):
    """ Crea la numeriacion de los triangulos en el nivel level """
    sup_trian = []
    inf_trian = []
    level = level*nb
    for i in range(aux_nb):
        b = [2*i+2+level, 2*i+1+level, 2*i+level]
        if i == aux_nb-1:
            b[0] = level
        c = list(b)
        c[0],c[1],c[2] = c[0]+nb,c[1]+nb,c[2]+nb
        b1 = list(b)
        c1= list(c)
        T1 = [b1[2],b1[1],c1[2]]
        T2 = [b1[1],b1[0],c1[0]]
        T3 = [c1[0],c1[1],b1[1]]
        T4 = [c1[1],c1[2],b1[1]]
        sup_trian.append(T1)
        sup_trian.append(T2)
        inf_trian.append(T3)
        inf_trian.append(T4)
    level_trian = sup_trian + inf_trian
    return level_trian

def triangles_lowest_level():
    """Agrega los elementos del ultimo nivel unidos al origen"""
    begin_index = (number_circles)*nb
    cen_index = (number_circles+1)*nb
    last_triangles = []
    for i in range(nb):
        r, s = begin_index+i, begin_index+i+1
        if i is nb-1:
            s = begin_index
        T = [r,s, cen_index]
        last_triangles.append(T)
    return last_triangles

# Este loop agrega los triangulos de los anillo mas los del que inciden
# en el origen

triangles = []
for i in range(number_circles):
    triangles =  triangles + triangles_level(i)

triangles = triangles + triangles_lowest_level()

#------------------------------------------------------------------------------
#                            Grafica de la malla
#------------------------------------------------------------------------------

plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(x, y, triangles, 'go-')
plt.title('Malla version 4')
#plt.xlabel('Longitude (degrees)')
#plt.ylabel('Latitude (degrees)')
plt.gca().set_xlim([-1.5-radius,radius+1.5])
plt.gca().set_ylim([-1.5-radius,radius+1.5])
plt.show()
