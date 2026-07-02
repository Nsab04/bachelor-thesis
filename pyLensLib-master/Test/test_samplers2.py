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