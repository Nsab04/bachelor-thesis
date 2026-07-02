from pyLensLib.piemd import piemd
from pyLensLib.sersic import *
from pyLensLib.observation import observation
import numpy as np
from astropy.cosmology import FlatLambdaCDM

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

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

npix=1000
pixsize = 0.01
sz =  pixsize * npix
theta=np.linspace(-sz/2.,sz/2.,1000)

dpie.setGrid(theta)

import matplotlib.pyplot as plt

#fig,ax = plt.subplots(1,1,figsize=(10,10))
#ax.plot(dpie.theta1,dpie.theta2,',',color='red')
#plt.show()


tl=dpie.tancl()
rl=dpie.radcl()

#for t in tl:
#    x,y = dpie.getCritPoints(t)
#    ax.plot(x,y,'-')

#for r in rl:
#    x,y = dpie.getCritPoints(r)
#    ax.plot(x,y,'--')
#plt.show()


mag_gal = 23.0
mag_len = 19.0
zp = 24.0
mag_sky = 22.0
ob = observation(size=sz, Npix=npix, zp=zp, texp=2000, bkg=mag_sky)

fl_gal = ob.mag2counts(mag_gal)
fl_len = ob.mag2counts(mag_len)

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

se = sersic(size=sz, Npix=npix,gl=dpie,save_unlensed=True,**kwargs)


kwargs_lens={
    'n': 1.0,
    're': 0.1,
    'q': 0.6,
    'pa': -np.pi / 4.0,
    'ys1': 0.0,
    'ys2': 0.0,
    'flux': fl_len,
    'zs': zl
}

se_lens = sersic(size=sz, Npix=npix,gl=None,**kwargs_lens)


noise=ob.makeNoise(se.image+se_lens.image)


fig,ax = plt.subplots(1,3,figsize=(27,10))
ax[0].imshow(se.image_unlensed,origin='lower')
ax[1].imshow(se.image,origin='lower')
toshow=se.image+se_lens.image+noise
ax[2].imshow(toshow,origin='lower',vmax=toshow.max()*0.05)
plt.show()