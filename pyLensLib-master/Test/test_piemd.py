from pyLensLib.piemd import piemd
from pyLensLib.sersic import *
import numpy as np
from astropy.cosmology import FlatLambdaCDM


import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pyLensLib.observation import observation

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

r=np.logspace(-2, np.log10(3.0), 100)

thetat=[1000000,1000,500]
thetac=[1, 10, 50]

for i in range(len(thetat)):
    kwargs = {'zl': 0.5,
              'zs': 2.0,
              'sigma0': 1000.0,
              'q': 1.0,
              'pa': -np.pi / 4.0,
              'theta_c': 10.0,
              'theta_t': thetat[i],
              'x1': 0.0,
              'x2': 0.0}
    dpie = piemd(co, **kwargs)
    m=dpie.m3Dr(r)
    v=dpie.vcirc(r)
    plt.plot(r, v, '-',color='b')

for i in range(len(thetat)):
    kwargs = {'zl': 0.5,
              'zs': 2.0,
              'sigma0': 1000.0,
              'q': 1.0,
              'pa': -np.pi / 4.0,
              'theta_c': thetac[i],
              'theta_t': 1000.0,
              'x1': 0.0,
              'x2': 0.0}
    dpie = piemd(co, **kwargs)
    m = dpie.m3Dr(r)
    v = dpie.vcirc(r)
    plt.plot(r,v,'--',color='r')

#plt.yscale('log')
#plt.ylim([1e13,1e15])
plt.ylim([1000,1500])
plt.show()



zl=0.5
zs=1.0

kwargs = {'zl': zl,
          'zs': zs,
          'sigma0': 170.0,
          'q': 0.6,
          'pa': -np.pi / 4.0,
          'theta_c': 0.001,
          'theta_t': 3.0,
          'x1': 0.0,
          'x2': 0.0}

dpie=piemd(co,**kwargs)


mag_gal=23.0
mag_lens=18.0
mag_sky=22.5
zp=23.9

npix=300
sz=0.01*npix

ob = observation(size=sz, Npix=npix, zp=zp, texp=2000, bkg=mag_sky)

fl_gal=ob.mag2counts(mag_gal)
fl_lens=ob.mag2counts(mag_lens)
sky_counts=ob.bkg_counts

kwargs={
    'n': 1.0,
    're': 0.1,
    'q': 1.0,
    'pa': np.pi/4.,
    'ys1': 0.1,
    'ys2': 0.0,
    'flux': fl_gal,
    'zs': zs
}

kwargs_lens={
    'n': 4.0,
    're': 1.5,
    'q': 0.6,
    'pa': -np.pi/4.,
    'ys1': 0.0,
    'ys2': 0.0,
    'flux': fl_lens,
    'zs': zl
}

theta=np.linspace(-sz/2.,sz/2.,1000)
dpie.setGrid(theta)

tl=dpie.tancl()
rl=dpie.radcl()

"""
deta=(1.0-dpie.ka)**2-dpie.g1**2-dpie.g2**2
plt.imshow(deta,extent=[-sz/2.,sz/2.,-sz/2.,sz/2.],origin='low')
for c in tl:
    x1,x2 = dpie.getCritPoints(c)
    plt.plot(x1,x2,'-')
for c in rl:
    x1,x2 = dpie.getCritPoints(c)
    plt.plot(x1,x2,'-')
plt.show()
"""
se=sersic(size=sz, Npix=npix,gl=dpie,**kwargs)
se_lens=sersic(size=sz, Npix=npix,gl=None,**kwargs_lens)
image=se.image#/se.image.sum()*fl_gal

"""
import noise
import numpy as np
from scipy.misc import toimage

shape = (npix, npix)
scale = 100.0
octaves = 10
persistence = 0.9
lacunarity = 2.0

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i / scale,
                                    j / scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=npix,
                                    repeaty=npix,
                                    base=0)

#toimage(world).show()
#plt.imshow(world)
#plt.show()
image=image*(1.0+world)
"""





image_lens=se_lens.image


px=sz/npix


print (zp-2.5*np.log10(image.sum()))
noise=ob.makeNoise(image+image_lens)
print (noise.mean(),image.max(),image.mean())
# SNR=1 level, calculated by solving S/sqrt(S+B)=1
thresh=3.0
snr = 0.5*(1.0+np.sqrt(1.0+4.0*ob.bkg_counts*ob.texp))/ob.texp*thresh



fig,ax=plt.subplots(1,5,figsize=(40,8))
ax[0].imshow((image+image_lens+noise),origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.],vmax=(image+image_lens+noise).max()*0.2)

"""
for c in tl:
    x1,x2 = dpie.getCritPoints(c)
    ax[0].plot(x1,x2,'-',color='yellow')
for c in rl:
    x1,x2 = dpie.getCritPoints(c)
    ax[0].plot(x1,x2,'-')
"""

ax[1].imshow(np.sqrt(image),origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
cnt=se.image_contours(level=1.0*snr)
cnt_lens=se_lens.image_contours(level=1.0*snr)
for c in cnt:
    x1,x2 = se.getContourPoints(c,px)
    ax[1].plot(x1,x2,'--',color='red')

ax[2].imshow(np.log10(image_lens),origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
for c in cnt_lens:
    x1,x2 = se_lens.getContourPoints(c,px)
    ax[2].plot(x1,x2,'--',color='red')

above_snr = image > snr
segm = np.zeros(image.shape)
segm[above_snr] = 1.0
ax[3].imshow(segm,origin='lower')

above_snr = image_lens > snr
segm = np.zeros(image_lens.shape)
segm[above_snr] = 1.0
ax[4].imshow(segm,origin='lower')

plt.show()



#import h5py
#hf = h5py.File('/Users/massimo/stiva/HUDF/hudf_dataset.h5', 'r')
#gals=hf.get('hudf_resized').value


import numpy as np
from math import *

n=1000
a=0.5
b=0.6
th=np.random.randn(n)
x=a*np.exp(b*th)*np.cos(th)
y=a*np.exp(b*th)*np.sin(th)
x1=a*np.exp(b*(th))*np.cos(th+pi)
y1=a*np.exp(b*(th))*np.sin(th+pi)

sx=np.random.normal(0, a*0.25, n)
sy=np.random.normal(0, a*0.25, n)
plt.plot(x+sy,y+sx,"*")
plt.plot(x1+sx, y1+sy,"*")

plt.show()