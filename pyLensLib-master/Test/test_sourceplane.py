from pyLensLib.sourceplane import sersic_sourceplane
from pyLensLib.nfwell import nfwell
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
co = FlatLambdaCDM(H0=70.0, Om0=0.3)

# create a source plane at z=zs with nrsc sources. The FOV is fsize^2 and the number of pixels is Npix^2
nsrc=50
zs=2.0
fsize=20.0
Npix=2000

# generate source properties
n = np.random.rand(nsrc)*3.0+1.0 # sersic index
re = np.random.rand(nsrc)*2.0+0.05 # effective radius
ys1 = (np.random.rand(nsrc)-0.5)*fsize # y1 coordinate on the source plane
ys2 = (np.random.rand(nsrc)-0.5)*fsize # Y2 coordinate on the source plane
fl = np.random.rand(nsrc)*10.0 # source fluxes (arbitrary units)
pa = np.random.rand(nsrc)*np.pi # position angle
q = np.random.rand(nsrc)*0.7+0.3 # axis ratios

# create an instance of an NFW lens
kwargs = {'zl': 0.5,
          'zs': zs,
          'mass': 2e13,
          'q': 0.7,
          'conc': 30.0,
          'pa': 0.0,
          'x1': 0.0,
          'x2': 0.0}

nfw = nfwell(co, **kwargs)
theta = np.linspace(-fsize/2., fsize/2., Npix)
nfw.setGrid(theta)

# create the instance of the sersic_sourceplane class, applying the lensing effects of the nfw lens
srcdict = {'nsrc': len(n), 'n': n, 're': re, 'ys1': ys1, 'ys2': ys2, 'flux': fl, 'pa': pa, 'q': q}
ss = sersic_sourceplane(size=fsize,Npix=Npix,save_unlensed=True,gl=nfw,rmaxf=10,**srcdict)

# show unlensed and lensed source plane
fig,ax=plt.subplots(1,2,figsize=(18,10))
ax[0].imshow(ss.ps.image_unlensed,origin='lower',vmax=0.001)
ax[1].imshow(ss.ps.image,origin='lower',vmax=0.001)
plt.show()
