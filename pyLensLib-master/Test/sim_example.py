"""
test script to generate a simulated observation of a galaxy lensed by another galaxy
- lens mass: the lens in this example is modeled with a PIEMD, whose parameters are set in the dictionary kwargs
- lens light: the lens light is modeled with a Sersic profile. Parameters are set in kwargs_lens_light
- source: the source in this example is modedeled with a Sersic profile. Parameters are set in kwargs_source_light
"""


from pyLensLib.piemd import piemd
from pyLensLib.sersic import *
import numpy as np
from astropy.cosmology import FlatLambdaCDM

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pyLensLib.observation import observation
import astropy.io.fits as fits
from skimage.transform import resize


# set up cosmology
co = FlatLambdaCDM(H0=70.0, Om0=0.3)

# lens and source redshifts
zl=0.5
zs=1.0

# properties of the lens
kwargs = {'zl': zl, # redshift
          'zs': zs, # source redshift (all lensing properties are computed for a source at this redshift
          'sigma0': 230.0, # lens central velocity dispersion
          'q': 0.6, # lens axis ratio b/a
          'pa': -np.pi / 4.0, # lens position angle
          'theta_c': 0.001, # lens core radius
          'theta_t': 3.0, # lens truncation radius
          'x1': 0.0, # lens position x
          'x2': 0.0} # lens position y

# create an instance of the lens
dpie=piemd(co,**kwargs)

# now deal with the light distributions
mag_source=23.0 # source magnitude
mag_lens=18.0 # lens magnitude
mag_sky=22.5 # sky brightness (mag/sq. arcsec)
zp=23.9 # zeropoint

# properties of the produced image
npix = 50 # pixels along both axes
pixscale = 0.1 # pixel scale in arcsec
sz=pixscale*npix # size of the postage stamp in arcsec
pixscale_highres = pixscale/10. # set pixel scale for higher resolution image
npix_highres = int(sz/pixscale_highres) # number of pixels in the higher resolution image

# create and instance of the observation
ob = observation(size=sz, Npix=npix_highres, zp=zp, texp=565*4, bkg=mag_sky)

# convert magnitudes to counts/s
fl_gal=ob.mag2counts(mag_source)
fl_lens=ob.mag2counts(mag_lens)
sky_counts=ob.bkg_counts

# properties of the light distributions
kwargs_source_light={
    'n': 1.0, # sersic index
    're': 0.1, # effective radius
    'q': 1.0, # axis ratio
    'pa': np.pi/4., # position angle
    'ys1': 0.1, # position x (on the source plane)
    'ys2': 0.0, # position y (on the lens plane)
    'flux': fl_gal, # flux of the source (in counts/s)
    'zs': kwargs['zs']
}

kwargs_lens_light={
    'n': 4.0, # sersic index
    're': 2.5, # effective radius
    'q': kwargs['q'],
    'pa': kwargs['pa'],
    'ys1': kwargs['x1'],
    'ys2': kwargs['x2'],
    'flux': fl_lens,
    'zs': kwargs['zl']
}

# set up a grid to compute the lensing maps
npix_grid = 1000
theta=np.linspace(-sz/2.,sz/2.,npix_grid)
dpie.setGrid(theta)

# create sersic instances of the lens and of the source

se=sersic(size=sz, Npix=npix_highres,gl=dpie,**kwargs_source_light)
se_lens=sersic(size=sz, Npix=npix_highres,gl=None,**kwargs_lens_light)

print (se_lens.image.sum(),fl_lens)

# compute the noise map
noise=ob.makeNoise(se.image+se_lens.image)

# compute the brightness level corresponding to a give SNR threshold
SNR = 1.0
noise_level = 0.5*(1.0+np.sqrt(1.0+4.0*ob.bkg_counts*ob.texp))/ob.texp*SNR

#  compute image threshold level
cnt=se.image_contours(level=noise_level)
cnt_lens=se_lens.image_contours(level=noise_level)

fig,ax=plt.subplots(2,5,figsize=(40,8))
image = se.image+se_lens.image+noise
ax[0,0].imshow(image,origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.],vmax=image.max()*0.2)

ax[0,1].imshow(np.sqrt(se.image),origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
for c in cnt:
    x1,x2 = se.getContourPoints(c,pixscale_highres)
    ax[0,1].plot(x1,x2,'--',color='red')

ax[0,2].imshow(np.log10(se_lens.image),origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
for c in cnt_lens:
    x1,x2 = se_lens.getContourPoints(c,pixscale_highres)
    ax[0,2].plot(x1,x2,'--',color='red')

above_snr = se.image > noise_level
segm = np.zeros(se.image.shape)
segm[above_snr] = 1.0
ax[0,3].imshow(segm,origin='lower')

above_snr = se_lens.image > noise_level
segm = np.zeros(se_lens.image.shape)
segm[above_snr] = 1.0
ax[0,4].imshow(segm,origin='lower')

# import PSF from an image

hdul = fits.open('/Users/massimo/share/astro/PSF/SystemPSF_reference090622_Euclid_spider3_lbda=800nm_pixelsize=0.6um=5e-3asec_FWHM=0.18asec_gaussian_enlargement.fits', memmap=False)
psf_image = hdul[0].data
px_psf = 5e-3#hdul[0].header['CD2_2']

# convolve image with PSF

image_convolved = ob.convolve_psf(se.image+se_lens.image, psf_image, px_psf)

# resize to match the desired resolution
image_resized = resize(image_convolved,(npix, npix),anti_aliasing=True,preserve_range=True)
image_resized = image_resized/image_resized.sum()*image_convolved.sum()

# re-observe at lower resolution
ob_lowres = observation(size=sz, Npix=npix, zp=zp, texp=565*4, bkg=mag_sky)
noise_lowres=ob_lowres.makeNoise(image_resized)

# show results of convolution
image_to_display = image_resized+noise_lowres
ax[1,0].imshow(image_convolved,vmax=image_convolved.max()*0.9,origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
ax[1,1].imshow(image_resized,vmax=image_resized.max()*0.9,origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
ax[1,2].imshow(image_to_display,vmax=image_to_display.max()*0.9,origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
plt.show()
