import matplotlib.pyplot as plt
import numpy as np
import sys, csv, os


from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.colors import LightSource
import random

if 'linux' in sys.platform:
    directory = 'acquisition/data/big_metasurface'
else:
    directory = 'acquisition\\data\\small_metasurfaces\\angles\\top4left1'

subdirs = ['order-2', 'order-1']
#angledirs = ['-40dgr','-38dgr','-36dgr','-34dgr','-32dgr','-30dgr','-28dgr','-26dgr','-24dgr','-22dgr','-20dgr','-18dgr','-16dgr','-14dgr','-12dgr','-10dgr','-8dgr','-6dgr','-4dgr','-2dgr','0dgr', '2dgr','4dgr','6dgr','8dgr','10dgr','12dgr','14dgr','16dgr','18dgr','20dgr','22dgr','24dgr','26dgr','28dgr','30dgr','32dgr','34dgr','36dgr','38dgr','40dgr']
angledirs = ['-40dgr','-36dgr','-32dgr','-28dgr','-24dgr','-20dgr','-16dgr','-12dgr','-8dgr','-4dgr',
             '0dgr','4dgr','8dgr','12dgr','16dgr','20dgr','24dgr','28dgr','32dgr','36dgr','40dgr']
os.chdir(directory)

####################################################################
#calculating efficiency
##try:
##    with open('powers.txt') as f:
##        raw = f.readlines()
##
##    efficiency = 0.
##        
##    for i in range(len(raw)):
##        raw[i] = raw[i].split(' ')
##        if 'mW' in raw[i][-1]:
##            raw[i][1] = float(raw[i][1])*1000
##        if 'inc' in raw[i][0]:
##            inc_power = float(raw[i][1])
##        if raw[i][0]!='0:' and raw[i][0]!='inc:':
##            efficiency += float(raw[i][1])
##            
##    efficiency = efficiency/inc_power
##    print('4-orders efficiency:',efficiency)
##except:
##    pass
####################################################################
#calculating efficiency

data=[]
data_thorlabs=[]
for folder in angledirs:
    os.chdir(folder)
    os.chdir('order-1')
    with open('polarimeter.txt') as f:
        polarimeter = np.array(list(csv.reader(f)), dtype='float')
        polarimeter = np.array([ 1 ] + list(np.mean(polarimeter,0)[0:3]))
        data_thorlabs.append(polarimeter)

##    try:
##        data.append(np.hstack([pol_only.transpose()[1],qwp_R.transpose()[1],qwp_L.transpose()[1]]))
##    except:
##        pass
    os.chdir('..')
    os.chdir('..')
    
    
data = np.array(data)    
data_thorlabs = np.array(data_thorlabs)

##A = [[1,1,0,0],
##     [1,0,1,0],
##     [1,-1,0,0],
##     [1,0,-1,0],
##     [1,0,0,1],
##     [1,0,0,-1]]
##A = np.array(A)
##Ainv = np.linalg.pinv(A)

##measured_stokes=np.zeros((4,4))
##for i in range(len(data)):
##   measured_stokes[i] = np.dot(Ainv, data[i])
##   measured_stokes[i][0] = 1.
##   measured_stokes[i][1:] = measured_stokes[i][1:] / np.linalg.norm(measured_stokes[i][1:])
##

#############################################################################
# plotting on Poincare sphere

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# Set the aspect ratio to 1 so our sphere looks spherical
fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(111, projection='3d')

def plot_sphere(ax,arrows='xyz',equatorial=True):
    phi = np.linspace(0, np.pi, 200)
    theta = np.linspace(0, 2*np.pi, 200)

    #equatorial circle
    xe=np.sin(theta)
    ye=np.cos(theta)

    phi, theta = np.meshgrid(phi, theta)

    # The Cartesian coordinates of the unit sphere
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    ax.plot_surface(x, y, z,  rstride=10, cstride=10, color='#EBE3E8',
                antialiased=True, alpha=0.5, lw=0.)#, facecolors=cm)
    if 'y' in arrows:
        ax.add_artist(Arrow3D([0, 0], [-0.03, 1.5], 
                        [0,0], mutation_scale=15, 
                        lw=0.25, arrowstyle="-|>", color="black"))
        ax.text(0,1.5,0, '$S_2$', fontweight='bold')        
    if 'x' in arrows:
        ax.add_artist(Arrow3D([0.0, 1.5], [0,0], 
                        [0,0], mutation_scale=15, 
                        lw=0.25, arrowstyle="-|>", color="black"))
        ax.text(1.6,0,0, '$S_1$', fontweight='bold')        
    if 'z' in arrows:        
        ax.add_artist(Arrow3D([0, 0], [0,0], 
                        [-0.03,1.5], mutation_scale=15, 
                        lw=0.25, arrowstyle="-|>", color="black"))
        ax.text(0,0,1.5, '$S_3$',fontweight='bold')
    if equatorial:
        ax.plot(xe,ye,0,'--', dashes=(10, 10), lw=0.25, color='red', alpha=1)

plot_sphere(ax)

# Plotting Thorlabs polarimeter data
S1 = data_thorlabs.transpose()[1]
S2 = data_thorlabs.transpose()[2]
S3 = data_thorlabs.transpose()[3]

for i in range(0,4):
    for j in range(0,4):
        plt.plot(list(S1[n] for n in [i,j]),
                 list(S2[n] for n in [i,j]),
                 list(S3[n] for n in [i,j]), color='orange', lw=0.5, marker=' ')

### Plotting measured polarization data
##S1 = measured_stokes.transpose()[1]
##S2 = measured_stokes.transpose()[2]
##S3 = measured_stokes.transpose()[3]
##for i in range(0,4):
##    for j in range(0,4):
##        plt.plot(list(S1[n] for n in [i,j]),
##                 list(S2[n] for n in [i,j]),
##                 list(S3[n] for n in [i,j]), color='blue', lw=0.5, marker=' ')

# Turn off the axis planes
#vinkler = np.arange(-40,41,2)
vinkler = np.arange(-40,41,4)
#vinkler = np.insert(vinkler,0, [-40,-36,-32,-28,-24])
#vinkler = np.insert(vinkler,len(vinkler), [24,28,32,36,40])
ax.set_axis_off()
plt.show()
#plt.plot(vinkler, S1,'-o')
#plt.plot(vinkler, S2,'-o')
#plt.plot(vinkler, S3,'-o')
plt.scatter(vinkler, S1)
plt.scatter(vinkler, S2)
plt.scatter(vinkler, S3)
plt.show()

