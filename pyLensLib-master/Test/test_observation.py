"""
This script is aimed at testing the observation module, which contains the observation class.
It uses also the sersic, sersic2, lenstool, and deflector modules
"""

"""
Step 1: create a deflector instance using the lenstool module

The lenstool module of pyLensLib contains some methods to deal with the Lenstool outputs.
Among them, the create_deflector function can be used to generate an instance of the deflector 
class from lenstool deflection angle maps and parameter files.

"""

import pyLensLib.lenstool as lst
from pyLensLib.sersic2 import sersic2
from pyLensLib.sersic import sersic
from pyLensLib.observation import observation
import numpy as np

from icecream import ic

############ Interface to observed cluster database

# cluster nicknames
cluster = ['S1063','M0416','M1206pl','PSZ1G311_200',
           'A370','A2744','M0717','M1149',
           'M0329','M1931','M2129','R2129']

# paths to the deflection angle and parameter files
path_to_angles = ['/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/stiva/clusters/PSZ1G311/delens/',
                  '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/massimo/stiva/pietro_models/',
                  '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/']


#
zl_ = [0.439,0.396,0.348,0.4436,0.375,0.308,0.545,0.543,0.45,0.352,0.587,0.234]

icl = 1
zs = 7
print (('Working with cluster %s') % (cluster[icl]))
print ('creating deflector...')

df = lst.create_deflector(parfile = path_to_angles[icl] + cluster[icl] + '.par',
                          filex = path_to_angles[icl] + cluster[icl] + '_angx.fits',
                          filey = path_to_angles[icl] + cluster[icl] + '_angy.fits',
                          zl= zl_[icl], zs = zs, zsnorm = 1.0, resc_fact = 1.0)

lims = lst.getFoV(path_to_angles[icl] + cluster[icl] + '.par')
fov = lims[1]-lims[0]

print (("FOV= %4.1f x %4.1f") % (fov,fov))

#import matplotlib.pyplot as plt
#fig,ax =plt.subplots(1,1,figsize=(10,10))
#gamma = np.sqrt(df.g1**2+df.g2**2)
#detA = (1.0-df.ka)**2-gamma**2
#ax.imshow(abs(1./detA),origin='lower',vmax=100.0)
#plt.show()

theta1 = 10.0
theta2 = 5.0

a1, a2 = df.getAngle(theta1,theta2)
beta1 = theta1 - a1
beta2 = theta2 - a2

ic (beta1, beta2)

"""
Step 2: create an observation instance

The observation class contains methods to simulate observations with a give instruments. The user must specify
the FOV of the observation, the number of pixels on each side, the instrument zero-point, the limiting magnitude 
for point sources in a circular aperture of radius rap to obtain a signal to noise sn
"""
Npix = 1000

ob = observation(size=fov, Npix=Npix, zp=23.9, mlim=24.5, rap=1.3, sn=10.0)

"""
Step 3: create a mock source with a Sersic brightness profile

pyLensLib contains two modules to deal with Sersic sources. The first is called sersic and the second sersic2. 
sersic2 is supposed to be faster (it is an interface to astropy functions). They work very similarly, but 
but there is a difference in the name of the parameters used to instantiate a sersic object. While sersic uses 
the total galaxy flux, sersic2 uses the surface brightness at the effective radius to normalize the brightness
profile.
"""

# create a dictionary with the parameters characterizing the source:

mag_src = 24.0

flux = ob.mag2counts(mag_src)

kwargs = {
    'n': 1.0, # sersic index
    'q': 0.6, # axis ratio
    'ys1': 5.0, # x position (in arcsec on the source plane)
    'ys2': 5.0,  # y position (in arcsec on the source plane)
    'pa': np.pi/4.0, # position angle (in radiants, counter-clockwise from the vertical axis)
    're': 1.0, # effective radius (in arcsec)
    'Ie': 100, # surface brightness at re (arbitrary units, can be omitted if using sersic
    'flux': flux, # total flux (in arbitrary units -- can be omitted if using sersic2)
    'zs': 1.0 # source redshift
}

se = sersic(size=fov, Npix=Npix, gl=df, save_unlensed=True, **kwargs)

from pyLensLib.pointsrc import pointsrc

kwargs_ps = {
    'ys1': 5.0,
    'ys2': 5.0,
    'flux': 1.0,
    'zs': 1.0
}
ps = pointsrc(size=fov, Npix=Npix, gl=df, **kwargs_ps)
#ps = pointsrc(size=5.0, Npix=Npix, gl=df, **kwargs_ps)
thetai1,thetai2,mui = ps.find_images()

import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,2,figsize=(18,10))
ax[0].imshow(se.image_unlensed,origin='lower',extent=[-fov/2.,fov/2,-fov/2.,fov/2])
ax[1].imshow(se.image,origin='lower',extent=[-fov/2.,fov/2,-fov/2.,fov/2])
ax[1].plot(thetai1,thetai2,'o',color='red')
plt.show()

"""
Step 4: observe the source

Now we can use the methods from the observation class to produce a mock observation of the source. 
"""

# read a PSF file
import astropy.io.fits as fits

hdul = fits.open('/Users/massimo/share/astro/PSF/HST_ACS_WFC_PSF/HST_ACS_WFC_F814W_PSF00.fits', memmap=False)
psf_image = hdul[0].data
px_psf = 0.0495

image_convolved = ob.convolve_psf(se.image, psf_image, px_psf)
image_unlensed_convolved = ob.convolve_psf(se.image_unlensed, psf_image, px_psf)

noise=ob.makeNoise(image_convolved)
noise_unlensed=ob.makeNoise(image_unlensed_convolved)


fig,ax = plt.subplots(1,2,figsize=(18,10))
ax[0].imshow(image_unlensed_convolved+noise_unlensed,origin='lower',extent=[-fov/2.,fov/2,-fov/2.,fov/2])
ax[1].imshow(image_convolved+noise,origin='lower',extent=[-fov/2.,fov/2,-fov/2.,fov/2])
ax[1].plot(thetai1,thetai2,'o',color='red')
plt.show()








