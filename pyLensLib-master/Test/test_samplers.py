from pyLensLib import samplers as sa
import matplotlib.pyplot as plt
import numpy as np

a0 = 1.0  # major axis length
b0 = 0.5  # minor axis length
p = sa.ellipsoidal_distribution(n=5000,saxes=[a0,b0])
print (p.shape)



fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.plot(p[:,0],p[:,1],'o',alpha=0.5,color='red')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])

# rotate the point distribution
theta_0 = 60.0
theta = np.deg2rad(theta_0)
ca = np.cos(theta)
sa = np.sin(theta)

rotM = np.array(
    [[ca,sa],[-sa,ca]]
)

p = np.matmul(p,rotM)

ax.plot(p[:,0],p[:,1],'o',alpha=0.5,color='blue')

#### now testing the method used in clusters
import sys
sys.path.append("/Users/massimo/MYPHYTHON")
from inertia_tensors import inertia_tensors

I = inertia_tensors(p)
evals, evecs = np.linalg.eigh(I)
evals = evals/np.max(evals)

# major and minor axis
a = np.sqrt(evals[0,1])
b = np.sqrt(evals[0,0])

# misalignment with cartesian axes
e1 = evecs[0,1,:]
e2 = evecs[0,0,:]

ux = np.array([[1.0,0.0]])
uy = np.array([[0.0,1.0]])
from rotations import  rotate_vector_collection
from rotations.rotations2d import rotation_matrices_from_angles
e1_0 = rotate_vector_collection(rotM,ux)[0]
e2_0 = rotate_vector_collection(rotM,uy)[0]

from rotations.vector_utilities import angles_between_list_of_vectors
misalignment_angle = np.degrees(angles_between_list_of_vectors(ux,e1))
misalignment_angle = np.minimum(misalignment_angle,180-misalignment_angle)

from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse

ellipse_0 = Ellipse([0,0], a0*2, b0*2, angle=theta_0,
                    edgecolor='red', lw=2, facecolor='none', linestyle='-')

ellipse_1 = Ellipse([0,0], a*2, b*2, angle=misalignment_angle,
                    edgecolor='red', lw=2, facecolor='none', linestyle='--')

ax.add_artist(ellipse_0)
ax.add_artist(ellipse_1)

inv_rotM=np.array([e1,e2])

p = np.matmul(p,inv_rotM.T)
print (p.shape)

ax.plot(p[:,0],p[:,1],'o',alpha=0.5,color='green')
plt.show()


####

from pyLensLib.sersic import sersic
from pyLensLib import samplers as sa
from pyLensLib.poststamp import poststamp
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import matplotlib.pyplot as plt
co = FlatLambdaCDM(H0=70.0, Om0=0.3)
import astropy.io.fits as fits
from PIL import Image



# set up source
kwargs = {
    'n': 1.0,
    'q': 0.2,
    'ys1': 1.0,  # ys1+dy1,
    'ys2': 1.0,  # ys2+dy2,
    'pa': np.pi/4.0,
    're': 2.0,
    'flux': 1.0,
    'zs': 1.0
}

npix=50
fsize =10.0
sizex = [-5., 5.]
sizey = [-5., 5.]
#se = sersic(size=fsize, Npix=npix, gl=None, save_unlensed=True,rmaxf=10.0, sizex=sizex, sizey=sizey, **kwargs)
se = sersic(size=fsize, Npix=npix, gl=None, save_unlensed=True,rmaxf=10.0, **kwargs)


l=np.linspace(0.1,1.0,1000)
fl = sa.PowerLawExpCut(l,delta=1.0,xcut=1.0)

fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.plot(l,fl,'-',label='1')
fl = sa.PowerLawExpCut(l,delta=1.0,xcut=0.5)
ax.plot(l,fl,'-',label='2')
fl = sa.PowerLawExpCut(l,delta=1.0,xcut=0.1,gamma=3)
ax.plot(l,fl,'-',label='3')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

#l = sa.samplePowerLawExCut(delta=1.0,xcut=0.2,gamma=5,nx=100000,xmin=0.5,fsub=1.0)
l_ = sa.samplePowerLawExCut(xmin=0.0001,fsub=1.0,delta=1.0,beta=-2.0,gamma=2.0)
print (np.sum(l_))

fig,ax= plt.subplots(1,1,figsize=(10,10))
ax.hist(l_,bins=10,density=True)
ll = np.linspace(0.0001,1.0,100)
p = sa.PowerLawExpCut(ll,delta=1.0,xcut=0.01,beta=-2.0,gamma=2.0)
ax.plot(ll,p,'-')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

l_ = np.array(l_)
isel = l_> 0.001
l = l_[isel]

print (l)
fig,ax=plt.subplots(1,1,figsize=(20,20))
ax.imshow(se.image,origin='lower')
image=se.image.copy()
image[image<(image.max()*0.05)]=0.0
x,y = sa.sample2Dimage(image,n=len(l))
ax.plot(x,y,'o',color='red',alpha=0.8)

plt.show()
