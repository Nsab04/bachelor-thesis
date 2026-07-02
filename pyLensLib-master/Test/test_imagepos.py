from pyLensLib.piemd import piemd
from pyLensLib.sersic import *
import numpy as np
from astropy.cosmology import FlatLambdaCDM

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pyLensLib.observation import observation

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

zl=0.5
zs=1.0

kwargs = {'zl': zl,
          'zs': zs,
          'sigma0': 170.0,
          'q': 0.8,
          'pa': -np.pi/4.0,
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
    'pa': 0.0,
    'ys1': 0.028,
    'ys2': 0.028,
    'flux': fl_gal,
    'zs': zs
}

theta=np.linspace(-sz/2.,sz/2.,1000)
dpie.setGrid(theta)

tl=dpie.tancl()
rl=dpie.radcl()

se=sersic(sizex=[-sz/2.,sz/2.], sizey=[-sz/2.,sz/2.], Npix=npix, gl=dpie, save_unlensed=True, **kwargs)
print (se.rescf)


fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(se.image,extent=[np.min(se.x1),np.max(se.x1),np.min(se.x2),np.max(se.x2)],origin='lower')
for t in tl:
    x,y=dpie.getCritPoints(t)
    ax.plot(x,y,'--',color='white')

selthetai=dpie.approximatedIP((kwargs['ys1'],kwargs['ys2']),dbetas=0.003)
#print (selthetai[0][0])

for i in range(selthetai.shape[0]):
    ax.plot(selthetai[i][1],selthetai[i][0],'o',alpha=0.8)

#isel = (se.y1-se.ys1)**2+(se.y2-se.ys2)**2 <= 0.01**2
#ax.plot(se.x1.flatten(),se.x2.flatten(),'+',color='orange')

ax.set_ylim([-sz/2-0.1,sz/2+0.1])
ax.set_xlim([-sz/2-0.1,sz/2+0.1])
plt.show()



