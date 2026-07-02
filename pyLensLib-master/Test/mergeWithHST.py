import numpy as np
import matplotlib.pyplot as plt
from acstools import acszpt
import astropy.io.fits as pyfits
import pickle
import pyLensLib.sedcompat as ss
from astropy.coordinates import SkyCoord, funcs
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb
from reproject import reproject_exact, reproject_interp
from skimage.util import random_noise
from pyLensLib.observation import observation
from matplotlib.backends.backend_pdf import PdfPages


def remove_sip(header):
    del header['A_*']
    del header['B_*']

    return header


def flux2counts(flux_hst, header_hst):
    '''
    Convert the flux F_nu into HST counts (e-/s)
    :param flux_hst: HST flux F_nu
    :param header_hst: HST header object
    :return: HST counts in e-/s
    '''

    abzp_hst = -2.5 * np.log10(header_hst['PHOTFLAM']) - 5 * np.log10(header_hst['PHOTPLAM']) - 2.4079

    """
    try:
        date = header_hst['DATE'].split('T')[0]
        detector = header_hst['DETECTOR']
        filter = header_hst['FILTER1']
    
        if filter == 'CLEAR1L':
            filter = header_hst['FILTER2']
        q = acszpt.Query(date=date, detector=detector, filt=filter)
        abzp_table = q.fetch()['ABmag'][0].value

        if np.abs(abzp_table - abzp_hst) > 0.001:

            print('WARNING: zp in the database differ from the zp computed from the header. Database: ' + str(abzp_table) +', header: '+str(abzp_hst))
    except:
        pass
    """

    cost = (abzp_hst + 48.60) / 2.5
    counts_hst = flux_hst * 10**cost

    return counts_hst

def sedplot(sed, ax, log=False, ymax=None, label='',**kwargs):
    fl = sed.fl
    if ymax is not None:
        fl = sed.fl / sed.fl.max() * ymax
    if log:
        ax.loglog(sed.wa, fl, label=label)
    else:
        ax.plot(sed.wa, fl, label=label)
    ax.set_xlabel('Wavelength ($\AA$)',fontsize=kwargs['fontsize'])
    ax.set_ylabel('Flux (ergs s$^{-1}$cm$^{-2}$ $\AA^{-1}$)',fontsize=kwargs['fontsize'])

def mag2flux(mag,sed,filtern,filters):
    passband = ss.Passband(filtern)
    sed.normalise_to_mag(mag, passband)
    flux = []
    for filtern_ in filters:
        passband_ = ss.Passband(filtern_)
        flux_=sed.calc_flux(passband_)
        flux.append(ss.flambda_to_fnu(passband_.effective_wa,flux_))
    return flux

def cutout_hdu(hdu,coords,fov=200.0):
    w=WCS(hdu.header)
    image_=hdu.data
    size = u.Quantity((fov, fov), u.arcsec)
    position = SkyCoord(coords, frame='fk5',unit="deg")
    cutout_ = Cutout2D(image_, position, size, wcs=w)
    w_up=cutout_.wcs
    header_ = w_up.to_header()
    hdu_hst = pyfits.PrimaryHDU(cutout_.data,header=header_)
    return hdu_hst


def linear(inputArray, scale_min=None, scale_max=None):
    """Performs linear scaling of the input numpy array.

    @type inputArray: numpy array
    @param inputArray: image data array
    @type scale_min: float
    @param scale_min: minimum data value
    @type scale_max: float
    @param scale_max: maximum data value
    @rtype: numpy array
    @return: image data array

    """
    #print("img_scale : linear")
    imageData = np.array(inputArray, copy=True)

    if scale_min == None:
        scale_min = imageData.min()
    if scale_max == None:
        scale_max = imageData.max()

    imageData = imageData.clip(min=scale_min, max=scale_max)
    imageData = (imageData - scale_min) / (scale_max - scale_min)
    indices = np.where(imageData < 0)
    imageData[indices] = 0.0
    indices = np.where(imageData > 1)
    imageData[indices] = 1.0

    return imageData


# read simulation datacube
datacube = 'ggsl_M0416.fits'
hdul = pyfits.open(datacube)

# read pickle file
catalog = 'ggsl_M0416.pkl'
with open(catalog, "rb") as f:
    dff = pickle.load(f)

# set up simulation
mag_int = np.linspace(24.0,30.0,10)
filtern_FF = ['BPZ/HST_ACS_WFC_F435W.res',
              'BPZ/HST_ACS_WFC_F606W.res',
              'BPZ/HST_ACS_WFC_F814W.res']#,
#              'BPZ/HST_WFC3_IR_F125W.res',
#              'BPZ/HST_WFC3_IR_F140W.res',
#              'BPZ/HST_WFC3_IR_F160W.res']

filtern = 'BPZ/HST_ACS_WFC_F814W.res'

path_to_imgs = '/Users/massimo/stiva/CLASH_FF_0416/'
input_imgs = ['hlsp_frontier_hst_acs-30mas_macs0416_f435w_v1.0_drz.fits',
              'hlsp_frontier_hst_acs-30mas_macs0416_f606w_v1.0_drz.fits',
              'hlsp_frontier_hst_acs-30mas_macs0416_f814w_v1.0_drz.fits']

psf_models = ['/Users/massimo/share/astro/PSF/HST_ACS_WFC_PSF/HST_ACS_WFC_F435W_PSF00.fits',
              '/Users/massimo/share/astro/PSF/HST_ACS_WFC_PSF/HST_ACS_WFC_F606W_PSF00.fits',
              '/Users/massimo/share/astro/PSF/HST_ACS_WFC_PSF/HST_ACS_WFC_F814W_PSF00.fits']

psf_images = []
psf_scales = []
for psf in psf_models:
    hdu_psf = pyfits.open(psf)
    psf_images.append(hdu_psf[0].data)
    psf_scales.append(hdu_psf[0].header['PIXSCALE'])

hdu = []
for i in range(len(input_imgs)):
    hduhstl=pyfits.open(path_to_imgs+input_imgs[i])
    hduhstl[0].header = remove_sip(hduhstl[0].header)
    hdu.append(hduhstl[0])

showexample = True
pdf_pages = PdfPages('ggsl_examples_withlens.pdf')

hdul_outR = pyfits.HDUList()
hdul_outG = pyfits.HDUList()
hdul_outB = pyfits.HDUList()

for i in range(len(hdul)):
    img = hdul[i].data
    ra = hdul[i].header['CRVAL1']
    dec = hdul[i].header['CRVAL2']
    fov = (img.shape[0]-1)*hdul[i].header['CD2_2']*3600.0
    coords_ = str(ra) + " " + str(dec)
    ob=observation(size=fov, Npix=img.shape[0], zp=24.0, texp=1)
    sed = dff.sed[i]

    cutout=[]
    for j in range(len(hdu)):
        cutout.append(cutout_hdu(hdu[j],coords_,fov=fov))

    #plt.imshow(cutout[0].data,origin='lower')
    #plt.show()
    repr_image, footprint = reproject_exact(hdul[i],
                                            cutout[0].header)  # assuming that the resolution in all bands is the same
    repr_image_conv = []
    for k in range(len(hdu)):
        repr_image_conv.append(ob.convolve_psf(repr_image, psf_images[k], psf_scales[k]))

    for j in [0]: #range(len(mag_int)):
        flux = mag2flux(mag_int[j],sed,filtern,filtern_FF)
        img_cnts = []

        for k in range(len(hdu)):
            #repr_image, footprint = reproject_exact(hdul[i], cutout[k].header)
            #repr_image_conv = ob.convolve_psf(repr_image,psf_images[k],psf_scales[k])
            img_cnts.append(flux2counts(repr_image_conv[k] * flux[k], hdu[k].header))


        for k in range(len(hdu)):
            img_cnts[k]=random_noise(img_cnts[k],mode='poisson')+cutout[k].data # not really correct... but OK!

        if showexample:
            fig, ax = plt.subplots(1, 2, figsize=(18, 9))
            R=linear(img_cnts[2],scale_min=0.0,scale_max=0.2)#scale_max=0.05)
            G=linear(img_cnts[1],scale_min=0.0,scale_max=0.2)#scale_max=0.05)
            B=linear(img_cnts[0],scale_min=0.0,scale_max=0.03)#0.01)
            #Z = make_lupton_rgb(R, G, B, stretch=0.1, Q=5)
            Z=np.dstack((R,G,B))
            ax[0].imshow(Z, origin='lower')
            kwargs = {'fontsize': 22}
            sedplot(sed, ax=ax[1], log=True, ymax=1., **kwargs)
            ax[0].xaxis.set_tick_params(labelsize=22)
            ax[0].yaxis.set_tick_params(labelsize=22)
            ax[1].xaxis.set_tick_params(labelsize=22)
            ax[1].yaxis.set_tick_params(labelsize=22)
            plt.tight_layout()
            pdf_pages.savefig(fig, dpi=150)
            plt.close(fig)

        if i == 0:
            hdul_outR.append(pyfits.PrimaryHDU(data=img_cnts[2],header=cutout[2].header))
            hdul_outG.append(pyfits.PrimaryHDU(data=img_cnts[1], header=cutout[1].header))
            hdul_outB.append(pyfits.PrimaryHDU(data=img_cnts[0], header=cutout[0].header))
        else:
            hdul_outR.append(pyfits.ImageHDU(data=img_cnts[2],header=cutout[2].header))
            hdul_outG.append(pyfits.ImageHDU(data=img_cnts[1], header=cutout[1].header))
            hdul_outB.append(pyfits.ImageHDU(data=img_cnts[0], header=cutout[0].header))

pdf_pages.close()
hdul_outR.writeto('allimagesR.fits',overwrite=True)
hdul_outG.writeto('allimagesG.fits',overwrite=True)
hdul_outB.writeto('allimagesB.fits',overwrite=True)
