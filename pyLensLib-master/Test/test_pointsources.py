
# the first point source is always at the center of the image
beta = [0.0,0.0]

from pyLensLib.pointsrc import pointsrc
from pyLensLib.observation import observation
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits

mag_source=24.5
Npix = 49
pixel_size = 0.1
fov = pixel_size*(Npix-1)
ob = observation(size=fov, Npix=Npix, zp=23.9, texp=565*4, mlim=24.5, rap=1.3, sn=10.0)
flux=ob.mag2counts(mag_source)

kwargs_psr = {
    'zs': 1.0,
    'ys1': beta[0],
    'ys2': beta[1],
    'flux': flux
}

mag_source2=24.0
beta2=[0.8,0.8]
kwargs_psr2 = {
    'zs': 1.0,
    'ys1': beta2[0],
    'ys2': beta2[1],
    'flux': flux
}



ps=pointsrc(size=fov, sizex=None, sizey=None, Npix=Npix, gl=None, **kwargs_psr)

ps2=pointsrc(size=fov, sizex=None, sizey=None, Npix=Npix, gl=None, **kwargs_psr2)

hdul = fits.open('/Users/massimo/share/astro/PSF/SystemPSF_reference090622_Euclid_spider3_lbda=800nm_pixelsize=0.6um=5e-3asec_FWHM=0.18asec_gaussian_enlargement.fits', memmap=False)
psf_image = hdul[0].data*1000.0

image = ps.image #+ ps2.image

from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
position = [(ps.image.shape[0]/2.0,ps.image.shape[1]/2.0)]
aperture = CircularAperture(position, r=1.3/ob.pixel)

image_conv = ob.convolve_psf(image, psf_image=psf_image, psf_scale=5e-3)
noise = ob.makeNoise(image_conv)
#fig,ax=plt.subplots(1,3,figsize=(18,10))
from matplotlib.colors import LogNorm
#ax[0].imshow(image+1,norm=LogNorm(),origin='lower')
#ax[1].imshow(image+1,norm=LogNorm(),origin='lower')
#ax[2].imshow(image_conv+noise+1,origin='lower',norm=LogNorm())

print ('Checking if noise level is well calibrated...')
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture, ApertureStats
from astropy.stats import SigmaClip
position = [(ps.image.shape[0]/2.0,ps.image.shape[1]/2.0),(14,14)]
aperture = CircularAperture(position, r=1.3/ob.pixel)
sigclip = SigmaClip(sigma=3.0, maxiters=10)
aperstats = ApertureStats(image_conv+noise+ob.bkg_counts, aperture,sigma_clip=sigclip)
phot_table = aperture_photometry(image_conv+noise+ob.bkg_counts, aperture)
phot_table['aperture_sum'].info.format = '%.8g'
nap = np.pi*(1.3/ob.pixel)**2

backg = aperstats.median[1]
signal = phot_table['aperture_sum'].value[0]-backg*nap

est_sn = signal/np.sqrt(signal+backg*nap)*np.sqrt(ob.texp)

print ('Estimated SNR', est_sn)
#print (phot_table)
#columns = ('id', 'mean', 'median', 'std', 'var', 'sum')
#stats_table = aperstats.to_table(columns)
#for col in stats_table.colnames:
#    stats_table[col].info.format = '%.8g'

#print (stats_table)


#plt.show()



mag_source_ini = 24.5
mag_source_ = np.linspace(20.0,mag_source_ini,10)
delta_mag_ = np.linspace(0,4,9)

d = np.linspace(0.1,0.5,9)
pa = np.random.random_sample(len(d)*len(mag_source_))*np.pi*2.0

fig,ax=plt.subplots(len(mag_source_),len(d),
                    figsize=(len(d)*2,len(mag_source_)*2))

j=0
ind = 0
for mag_source in mag_source_:
    flux=ob.mag2counts(mag_source)

    kwargs_psr = {
        'zs': 1.0,
        'ys1': beta[0],
        'ys2': beta[1],
        'flux': flux
    }
    ps = pointsrc(size=fov, sizex=None, sizey=None, Npix=Npix, gl=None, **kwargs_psr)
    for i in range(len(d)):
        flux2=flux
        print (j,i,mag_source,flux,d[i],np.rad2deg(pa[ind]))
        kwargs_psr2 = {
            'zs': 1.0,
            'ys1': beta[0]+d[i]*np.cos(pa[ind]),
            'ys2': beta[1]+d[i]*np.sin(pa[ind]),
            'flux': flux2
        }
        ps2=pointsrc(size=fov, sizex=None, sizey=None, Npix=Npix, gl=None, **kwargs_psr2)
        image = ps.image + ps2.image
        image_conv = ob.convolve_psf(image, psf_image=psf_image, psf_scale=5e-3)
        noise = ob.makeNoise(image_conv)
        ax[j,i].imshow(image_conv+noise+1,origin='lower',norm=LogNorm(),extent=[-fov/2,fov/2,-fov/2.,fov/2.])
        ax[j,i].set_title(('m$_{VIS}$=%4.1f d=%4.2f"') % (mag_source,d[i]),fontsize=10)
        ind+=1
    j+=1

plt.subplots_adjust(hspace=0.1)


fig.tight_layout()
plt.show()
fig.savefig('test.pdf',bbox_inches='tight',dpi=300)