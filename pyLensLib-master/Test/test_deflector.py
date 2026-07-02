import numpy as np

from pyLensLib.deflector import deflector
import astropy.io.fits as pyfits

potfile = '/Users/maxmen3/projects/Schools/Tonale/data/pot_D1_066_0.fits'
angfile = '/Users/maxmen3/projects/Schools/tmp/angles_D1_066_0.fits'

 # we import the module to manage .fits files from astropy.io. We call it pyfits.

hdul = pyfits.open(potfile) # open and read the file
pot = hdul[0].data # the data unit
header = hdul[0].header # the header unit

hdul1 = pyfits.open(angfile) # open and read the file
a1 = hdul1[0].data
a2 = hdul1[1].data

print (a1.max(),a2.max())

fov = header['NAXIS1']*header['CD2_2']*3600.0
pixelscale = header['CD2_2']*3600.0
npix = header['NAXIS1']

from astropy.cosmology import FlatLambdaCDM
co = FlatLambdaCDM(Om0=0.24,H0=72.0)
kwargs_def = {'zl': header['ZL'],'zs': header['ZS']}
df0 = deflector(co,pot=pot,usePotential=True,**kwargs_def)
df1 = deflector(co,angx=a1,angy=a2,**kwargs_def)

theta= np.linspace(-fov/2.,fov/2.,npix)
df0.setGrid(theta=theta)
df1.setGrid(theta=theta)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
fig,ax = plt.subplots(1,2,figsize=(18,10))
ax[0].imshow(df0.ka,vmax=2.0,vmin=0.0)
ax[1].imshow(df1.ka,vmax=2.0,vmin=0.0)
plt.show()


import lenstronomy.LensModel.convergence_integrals as len
a1_len,a2_len = len.deflection_from_kappa_grid(df0.ka,df0.pixel_scale)



fig,ax = plt.subplots(1,3,figsize=(24,10))
ax[0].plot(df0.theta1,df0.theta2,',',color='red')
ax[0].plot(df0.theta1-df0.a1,df0.theta2-df0.a2)
ax[1].plot(df1.theta1,df1.theta2,',',color='red')
ax[1].plot(df1.theta1-df1.a1,df1.theta2-df0.a2)
ax[2].plot(df1.theta1,df1.theta2,',',color='red')
ax[2].plot(df1.theta1-a1_len,df1.theta2-a2_len)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[2].set_aspect('equal')
plt.show()
