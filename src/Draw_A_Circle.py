import matplotlib.pyplot as plt
import matplotlib

plt.axes()

circle0 = plt.Circle((0, 0), radius=10., fc='b')

#H2
#circle1 = plt.Circle((5.0, 0), radius=4., fc='r', alpha = 0.75)

#H3
circle1 = plt.Circle((0.0, 0), radius=4., fc='r', alpha = 0.75)

#H3a
#circle1 = plt.Circle((0.0, 0), radius=0.9, fc='r', alpha = 0.75)

# HSquare
# rect1 = matplotlib.patches.Rectangle((5.-3.3,0.-3.3), 6.6, 6.6, color='r',alpha=0.75)

#H4
#circle1 = plt.Circle((5, 0), radius=4., fc='r', alpha = 0.75)
#circle2 = plt.Circle((-5, 0), radius=4., fc='r', alpha = 0.75)

#H5
#circle1 = plt.Circle((-8.0, 0), radius=1., fc='r', alpha = 0.75)
#circle2 = plt.Circle((-7.5, 3.5), radius=1., fc='r', alpha = 0.75)
#circle3 = plt.Circle((-7.5, -3), radius=1., fc='r', alpha = 0.75)
#circle4 = plt.Circle((7.5,0), radius=1., fc='r', alpha = 0.75)
#circle5 = plt.Circle((6.5, -5.5), radius=1., fc='r', alpha = 0.75)
#circle6 = plt.Circle((0, 0), radius=1., fc='r', alpha = 0.75)
#circle7 = plt.Circle((0, -7), radius=1., fc='r', alpha = 0.75)
#circle8 = plt.Circle((3.5, 5), radius=1., fc='r', alpha = 0.75)

#NH1
#circle1 = plt.Circle((0, 0), radius=5., fc='r', alpha = 0.75)
#circle2 = plt.Circle((0, 0), radius=4., fc='g', alpha = 0.95)

#NH2
# circle1 = plt.Circle((5, 0), radius=5., fc='r', alpha = 0.75)
# circle2 = plt.Circle((0, 0), radius=4., fc='g', alpha = 0.95)

plt.gca().add_patch(circle0)
#plt.gca().add_patch(rect1)
plt.gca().add_patch(circle1)
#plt.gca().add_patch(circle2)
#plt.gca().add_patch(circle3)
#plt.gca().add_patch(circle4)
#plt.gca().add_patch(circle5)
#plt.gca().add_patch(circle6)
#plt.gca().add_patch(circle7)
#plt.gca().add_patch(circle8)

plt.axis('scaled')

# Establece los limite
eps = 0.5
radius = 10.
limit = eps + radius
plt.xlim((-limit, limit ))
plt.ylim((-limit, limit ))

plt.show()