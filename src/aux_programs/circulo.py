import pylab
#cir1 = pylab.Circle((0,0), radius=0.75,  fc='y')
 #Creates a patch that looks like a circle (fc= face color)
cir1 = pylab.Circle((.75,.5), radius=0.125, alpha = .9 ,color = 'r') #
cir2 = pylab.Circle((.5,.5), radius=0.5, fc='b') 
#Repeat (alpha=.2 means make it very translucent)
ax = pylab.axes(aspect=1) #Creates empty axes (aspect=1 means scale things so that circles look like circles)
#ax.add_patch(cir1) #Grab the current axes, add the patch to it
ax.add_patch(cir2) #Repeat
ax.add_patch(cir1) #Repeat
pylab.show()

