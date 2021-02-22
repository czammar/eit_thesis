"""
Herramientas para la generacion de una malla

"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import meshpy.triangle as triangle

#####################################
#
# Genera puntos de un circulo que son vertices
# de un poligono regular inscrito
#
#####################################

def Circle(middle,radius,num_points=10,
                    a_min=0.,a_max=2.*np.pi,edge_length=-1):
  # check for closed loop
  number_points=num_points
  if edge_length>0:
    number_points=np.floor(abs(radius/edge_length*(a_max-a_min)))+1

  delta=(a_max-a_min)/number_points
  closed=False;
  if abs(a_max-a_min-2*np.pi)<0.1*delta:
    closed=True

  t=np.linspace(a_min,a_max,number_points,not closed)
  # define points
  points=[(middle[0]+radius*np.cos(angle),middle[1]+radius*np.sin(angle)) for angle in t]

  # define aristas
  aristas=[(j,j+1) for j in range(0,len(points)-1,1)]
  if closed==True:
    aristas+=[(len(points)-1,0)]
  return points, aristas;


#####################
#
# Generador de malla
#
####################

def DoTriMesh(points,vertices,edge_length=-1,holes=[],tri_refine=None,plot=None):
  info = triangle.MeshInfo()
  info.set_points(points)
  if len(holes)>0:
    info.set_holes(holes)
  info.set_facets(vertices)


  if tri_refine!=None:
    mesh = triangle.build(info,refinement_func=tri_refine)
  elif edge_length<=0:
    mesh = triangle.build(info)
  else:
    mesh = triangle.build(info,max_volume=0.5*edge_length**2)

  mesh_points = np.array(mesh.points)
  mesh_elements = np.array(mesh.elements)

  if plot == True:
      plt.gca().set_aspect('equal')
      plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_elements,'go-')
      plt.show()
  return mesh, mesh_points,mesh_elements;
