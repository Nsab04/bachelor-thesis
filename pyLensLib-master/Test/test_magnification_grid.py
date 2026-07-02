from pyLensLib.piemd import piemd
from pyLensLib.sersic import *
import numpy as np
from astropy.cosmology import FlatLambdaCDM

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pyLensLib.observation import observation
from pyLensLib.maputils import image_fit
from scipy.ndimage import map_coordinates

# set up cosmology
co = FlatLambdaCDM(H0=70.0, Om0=0.3)

# lens and source redshifts
zl = 0.5
zs = 1.0

# properties of the lens
kwargs = {'zl': zl,  # redshift
          'zs': zs,  # source redshift (all lensing properties are computed for a source at this redshift
          'sigma0': 1300.0,  # lens central velocity dispersion
          'q': 0.6,  # lens axis ratio b/a
          'pa': -np.pi / 4.0,  # lens position angle
          'theta_c': 10.0,  # lens core radius
          'theta_t': 1000.0,  # lens truncation radius
          'x1': 0.0,  # lens position x
          'x2': 0.0}  # lens position y

# create an instance of the lens
dpie = piemd(co, **kwargs)

# now deal with the light distributions
mag_source = 22.0  # source magnitude
mag_lens = 18.0  # lens magnitude
mag_sky = 22.5  # sky brightness (mag/sq. arcsec)
zp = 23.9  # zeropoint

# properties of the produced image
npix = 1000  # pixels along both axes
pixscale = 0.05  # pixel scale in arcsec
sz = pixscale * npix  # size of the postage stamp in arcsec

# set up a grid to compute the lensing maps
npix_grid = npix #1000
theta = np.linspace(-sz / 2., sz / 2., npix_grid)
dpie.setGrid(theta)

detA = (1.0 - dpie.ka) ** 2 - (dpie.g1 ** 2 + dpie.g2 ** 2)
mu=np.abs(1.0/detA)

# magnification on the source plane
y1 = dpie.theta1 - dpie.a1 + 2 * dpie.pixel_scale
y2 = dpie.theta2 - dpie.a2 + 2 * dpie.pixel_scale

import pymupds
mus = pymupds.mupds_triangle(y1 - dpie.theta1.min(), y2 - dpie.theta2.min(), dpie.pixel_scale, detA, nray=len(theta))

# create and instance of the observation
ob = observation(size=sz, Npix=npix, zp=zp, texp=565 * 4, bkg=mag_sky)

# convert magnitudes to counts/s
fl_gal = ob.mag2counts(mag_source)
fl_lens = ob.mag2counts(mag_lens)
sky_counts = ob.bkg_counts

# define a grid of sources
fov_zs = 20.0
thetas = np.linspace(-fov_zs/2.0,fov_zs/2.0,33)
thetas1, thetas2 = np.meshgrid(thetas,thetas)

thetas1=thetas1.flatten()
thetas2=thetas2.flatten()

s = []
from tqdm import tqdm
for i in tqdm(range(len(thetas1))):
    # properties of the light distributions
    kwargs_source_light = {
        'n': 1.0,  # sersic index
        're': 0.7,  # effective radius
        'q': 1.0,  # axis ratio
        'pa': np.pi / 4.,  # position angle
        'ys1': thetas1[i],  # 3.8,  # position x (on the source plane)
        'ys2': thetas2[i],  # 3.8,  # position y (on the lens plane)
        'flux': fl_gal,  # flux of the source (in counts/s)
        'zs': kwargs['zs']
    }
    # create sersic instances of the lens and of the source

    se = sersic(size=sz, Npix=npix, gl=dpie, save_unlensed=True, **kwargs_source_light)

    # compute the noise map
    noise = ob.makeNoise(se.image)

    # compute the brightness level corresponding to a give SNR threshold
    SNR = 1.0
    noise_level = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * ob.bkg_counts * ob.texp)) / ob.texp * SNR

    #  compute image threshold level
    cnt = se.image_contours(level=noise_level)
    cnt_source = se.source_contours(level=noise_level)

    # image centroid
    ai = image_fit(se.image, ith=noise_level)
    cen = ai.centroid()

    # estimate magnification from the mu map


    mask_lensed = se.image > noise_level
    mask_unlensed = se.image_unlensed > noise_level

    area_ratio = mask_lensed.sum() / mask_unlensed.sum()
    flux_ratio = np.sum(se.image * mask_lensed) / np.sum(se.image_unlensed * mask_unlensed)
    mean_mu = np.mean(mu[mask_lensed])
    median_mu = np.median(mu[mask_lensed])
    wfl_mu = np.average(mu[mask_lensed], weights=se.image[mask_lensed])
    mu_point = map_coordinates(mu, [[cen[1]], [cen[0]]], order=1, prefilter=True)

    mu_flux = np.sum((mus * se.image_unlensed) * mask_unlensed)
    flux = np.sum(se.image_unlensed * mask_unlensed)

    mu_true = mu_flux / flux

    s_entry = {'x': thetas1[i],'y': thetas2[i],
               'xi':cen[0]*pixscale-sz/2.0,'yi':cen[1]*pixscale-sz/2.,
               'mus_true': mu_true, 'area_ratio': area_ratio,
               'flux_ratio': flux_ratio, 'mean_mu': mean_mu,
               'median_mu': median_mu, 'wfl_mu': wfl_mu, 'mu_point': mu_point[0]}
    s.append(s_entry)

import pandas as pd
sdf=pd.DataFrame(s)

sdf.to_csv('source_mags.csv',sep=' ')


