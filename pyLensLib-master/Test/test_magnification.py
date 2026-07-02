from pyLensLib.piemd import piemd
from pyLensLib.sersic import *
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from skimage import measure
import shapely.geometry
from pyLensLib.critcau import CriticalLine
from shapely.ops import polygonize, unary_union

def sort_contours(contours):
    c_all = []
    j = 0
    for contour in contours:
        contour[:, 0], contour[:, 1] = contour[:, 1].copy(), contour[:, 0].copy()
        ls = shapely.geometry.LineString(contour)
        lr = shapely.geometry.LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(lr)
        mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
        c = CriticalLine(j, mp)
        c.setPoints(contour)
        A = 0.0
        for g in range(len(mp)):
            A += mp[g].area
        c.setArea(A)
        c_all.append(c)
        j += 1
    c_sorted = sorted(c_all, key=lambda x: x.getArea(), reverse=True)
    return (c_sorted)

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
npix = 2000  # pixels along both axes
pixscale = 0.025  # pixel scale in arcsec
sz = pixscale * npix  # size of the postage stamp in arcsec

# create and instance of the observation
ob = observation(size=sz, Npix=npix, zp=zp, texp=565 * 4, bkg=mag_sky)

# convert magnitudes to counts/s
fl_gal = ob.mag2counts(mag_source)
fl_lens = ob.mag2counts(mag_lens)
sky_counts = ob.bkg_counts

# properties of the light distributions
kwargs_source_light = {
    'n': 1.0,  # sersic index
    're': 1.0,  # effective radius
    'q': 1.0,  # axis ratio
    'pa': np.pi / 4.,  # position angle
    'ys1': 3.8,  # position x (on the source plane)
    'ys2': 3.8,  # position y (on the lens plane)
    'flux': fl_gal,  # flux of the source (in counts/s)
    'zs': kwargs['zs']
}

# set up a grid to compute the lensing maps
npix_grid = npix #1000
theta = np.linspace(-sz / 2., sz / 2., npix_grid)
dpie.setGrid(theta)

# create sersic instances of the lens and of the source

se = sersic(size=sz, Npix=npix, gl=dpie, save_unlensed=True, **kwargs_source_light)

# compute the noise map
noise = ob.makeNoise(se.image)

# compute the brightness level corresponding to a give SNR threshold
SNR = 1.0
#noise_level = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * ob.bkg_counts * ob.texp)) / ob.texp * SNR
#print (noise_level,ob.bkg_counts)
noise_level = 0.5 * SNR**2/ob.texp*np.sqrt(1+4.0*ob.texp/SNR**2*ob.bkg_counts)
#print (noise_level,ob.bkg_counts)
image = se.image + noise

#  compute image threshold level
cnt = se.image_contours(level=noise_level)
cnt_source = se.source_contours(level=noise_level)

# contours of noisy image
#cont_ = measure.find_contours(image, noise_level)
#noisy_cont = sort_contours(cont_)
#print (len(noisy_cont))

fig, ax = plt.subplots(3, 2, figsize=(20, 30))
ax[0, 0].imshow(image, origin='lower', extent=[-sz / 2., sz / 2., -sz / 2., sz / 2.], vmax=image.max() * 0.2)

ax[0, 1].imshow(np.sqrt(se.image), origin='lower', extent=[-sz / 2., sz / 2., -sz / 2., sz / 2.])

# magnification on the lens plane
detA = (1.0 - dpie.ka) ** 2 - (dpie.g1 ** 2 + dpie.g2 ** 2)
mu=np.abs(1.0/detA)

# magnification on the source plane
y1 = dpie.theta1 - dpie.a1 + 2 * dpie.pixel_scale
y2 = dpie.theta2 - dpie.a2 + 2 * dpie.pixel_scale

import pymupds
mus = pymupds.mupds_triangle(y1 - dpie.theta1.min(), y2 - dpie.theta2.min(), dpie.pixel_scale, detA, nray=len(theta))

#image centroid
ai = image_fit(se.image, ith= noise_level)
cen = ai.centroid()

# estimate magnification from the mu map
mu_point = map_coordinates(mu,[[cen[1]],[cen[0]]], order=1, prefilter=True)

for c in cnt:
    x1, x2 = se.getContourPoints(c, pixscale)
    ax[0, 1].plot(x1, x2, '--', color='red')
    ax[1, 0].plot(x1, x2, '--', color='red')
    ax[1, 1].plot(x1, x2, '--', color='red')

#for c in noisy_cont:
#x1, x2 = se.getContourPoints(noisy_cont[0], pixscale)
#ax[0, 1].plot(x1, x2, '-', color='white',)
#ax[1, 0].plot(x1, x2, '-', color='white')
#ax[1, 1].plot(x1, x2, '-', color='white')

for c in cnt_source:
    x1, x2 = se.getContourPoints(c, pixscale)
    ax[2, 0].plot(x1, x2, '--', color='red')
    ax[2, 1].plot(x1, x2, '--', color='red')

mask_lensed = se.image > noise_level
mask_unlensed = se.image_unlensed > noise_level

ax[1,0].imshow(mask_lensed,origin='lower',extent=[-sz / 2., sz / 2., -sz / 2., sz / 2.])
ax[1,0].plot(cen[0]*pixscale - sz/2.0,cen[1]*pixscale - sz/2.0,'o',color='red')


cl = dpie.tancl()
cau = dpie.getCaustics(cl)

for c in cl:
    x, y = dpie.getCritPoints(c)
    ax[0, 0].plot(x, y, '-', color='orange')
    ax[0, 1].plot(x, y, '-', color='orange')
    ax[1, 0].plot(x, y, '-', color='orange')
    ax[1, 1].plot(x, y, '-', color='orange')

for c in cau:
    x, y = dpie.getCausticPoints(c)
    ax[2, 0].plot(x, y, '-', color='orange')
    ax[2, 1].plot(x, y, '-', color='orange')



ax[1,1].plot(cen[0]*pixscale - sz/2.0,cen[1]*pixscale - sz/2.0,'o',color='red')
ax[1,1].imshow(np.log10(np.abs(mu))*mask_lensed,origin='lower',extent=[-sz / 2., sz / 2., -sz / 2., sz / 2.])


ax[2,0].imshow(mask_unlensed,origin='lower',extent=[-sz / 2., sz / 2., -sz / 2., sz / 2.])
ax[2,1].imshow(np.log10(np.abs(mus)),origin='lower',extent=[-sz / 2., sz / 2., -sz / 2., sz / 2.])
ax[2,1].grid(True)
fig.savefig('testmag.png')

print ('Magnification #1 (area ratio): %5.2f' % (mask_lensed.sum()/mask_unlensed.sum()))
print ('Magnification #2 (flux ratio): %5.2f' % (np.sum(se.image*mask_lensed)/np.sum(se.image_unlensed*mask_unlensed)))
print ('Magnification #3 (ext like - mean): %5.2f' % (np.mean(mu[mask_lensed])))
print ('Magnification #4 (ext like - median): %5.2f' % (np.median(mu[mask_lensed])))
print ('Magnification #5 (ext like - w mean): %5.2f' % (np.average(mu[mask_lensed],weights=se.image[mask_lensed])))
print ('Magnification #6 (point like): %5.2f' % (mu_point))
mu_flux = np.sum((mus*se.image_unlensed)*mask_unlensed)
flux = np.sum(se.image_unlensed*mask_unlensed)
#print (mu_flux,flux,mu_flux/flux)
print ('Magnification #7 (source plane flux integral): %5.2f' % (mu_flux/flux))


