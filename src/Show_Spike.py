from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy import sin, cos, pi
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#X = np.arange(0, 1, 1.0/51.0)
#Y = np.arange(0, 1, 1.0/51.0)
#X, Y = np.meshgrid(X, Y)
##Z = cos(pi*X) * sin(pi*Y) # solucion analitica de lines

#Z  = Y/( (1.0+X)**2 + Y**2)# solucion analitica de laplace
#ax.plot_surface(X, Y, Z,rstride=4, cstride=4, cmap='spectral')
#plt.show()

################################################################33

import matplotlib.tri as tri
import numpy as np

nb = 30 # Numero de nodos en la frontera (debe ser par)
aux_nb = (nb/2)
radius = 10. # radio del circulo
center = np.array([[0.0,0.0]]) # centro del circulo
number_circles = 3 # no. de circulos internos que tendra la malla 
radius_fraction = radius / number_circles
pi =np.pi

##################### Puntos de la malla
def PointsInCircum(r,n):
    """Genera en la circunferencia de radio r las coordenadas
        los vertices de un poligono regular de n lados"""
    l = [(np.cos(2*pi/n*x)*r, np.sin(2*pi/n*x)*r) for x in range(n)]
    l = np.array(l)
    return l

cloud_points = [ ]
#cloud_points.extend( PointsInCircum(radius, nb) )
#cloud_points.extend( PointsInCircum(radius * 0.75, nb) )
#cloud_points.extend( PointsInCircum(radius * 0.5, nb) )
#cloud_points.extend( PointsInCircum(radius * 0.25, nb) )
for i in range(number_circles+1):
    r= 1- (i /(number_circles+1.) )
    cloud_points.extend( PointsInCircum(radius*r, nb) )
cloud_points.extend(center)

cloud_points = np.array(cloud_points )
x= cloud_points[:,0]
y= cloud_points[:,1]

######### Triangulos de la malla

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

triangles = []
for i in range(number_circles):
    triangles =  triangles + triangles_level(i) 

triangles = triangles + triangles_lowest_level()

#################################3
#plt.figure()
#plt.gca().set_aspect('equal')
#plt.triplot(x, y, triangles, 'go-')
#plt.gca().set_xlim([-0.5-radius,radius+0.5])
#plt.gca().set_ylim([-0.5-radius,radius+0.5])
#plt.title('Malla Version IV')
##plt.xlabel('Longitude (degrees)')
##plt.ylabel('Latitude (degrees)')
#plt.show()


from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

# the function that I'm going to plot
def z_func(x,y):
 return (1-(x**2+y**3))*exp(-(x**2+y**2)/2)
 
x = arange(-3.0,3.0,0.1)
y = arange(-3.0,3.0,0.1)
X,Y = meshgrid(x, y) # grid of point
Z = z_func(X, Y) # evaluation of the function on the grid

im = imshow(Z,cmap=cm.RdBu) # drawing the function
# adding the Contour lines with labels
cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
# latex fashion title
title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
show()

### segunda parte

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

