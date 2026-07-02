import numpy as np
from pyLensLib.deflector import deflector
from pyLensLib.sersic import sersic
import matplotlib.pyplot as plt
from tqdm import tqdm
import astropy.io.fits as pyfits
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from pyLensLib.sedmodel import sedmodel
import pyLensLib.sedcompat as ss
import pickle
import pandas as pd

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches

sedm = sedmodel(sed_dir='/Users/massimo/CODES/bpz-1.99.3/SED/', sed_list='eB11.list')

filtern_FF = ['BPZ/HST_ACS_WFC_F435W.res',
              'BPZ/HST_ACS_WFC_F814W.res',
              'BPZ/HST_WFC3_IR_F105W.res',
              'BPZ/HST_WFC3_IR_F125W.res',
              'BPZ/HST_WFC3_IR_F140W.res',
              'BPZ/HST_WFC3_IR_F160W.res']

filtern = 'BPZ/HST_ACS_WFC_F814W.res'
magn = 20.0

simulation = True
ares = True

if (simulation and not ares):
    # assume to process several simulated clusters
    nclus = 1
    snap = '115'
    path_to_clusters = '/Users/massimo/stiva/300/SL/'
    npr = 3
    zl = None
    zs = None
    RA_REF = 0.0
    DEC_REF = 0.0
    cln = None
elif (simulation and ares):
    nclus = 1
    snap = ''
    path_to_clusters = ''
    cln = 'deflAnglesAres'
    zl = 0.5
    zs = 1.0
    RA_REF = 0.0
    DEC_REF = 0.0
    npr = 1
else:
    # process a real cluster
    nclus = 1
    snap = ''
    path_to_clusters = '/Users/massimo/stiva/pietro_models/shifted/'
    cln = 'M0416'
    npr = 1
    zl = 0.396
    zs = 1.0
    RA_REF = 64.038142
    DEC_REF = -24.067472

# show examples?
showexamples = True

# set cosmology
global co
co = FlatLambdaCDM(Om0=0.3, H0=70.0)

# number of arcs simulated for each iteration
narcs = 10
nzs = 20

# images
fov = 10.0
npix = 200

# select which lenses are interesting
minthetaE = 0.5
maxthetaE = 5.0

def remove_sip(header):
    del header['A_*']
    del header['B_*']

    return header

def readClusterDA(filealpha, simulation=True, ares=False, zl=None, zs=None):
    if (simulation and not ares):
        hdul = pyfits.open(filealpha)
        angx = hdul[0].data
        angy = hdul[1].data
        zl = hdul[0].header['ZL']
        zs = hdul[0].header['ZS']
        pix = hdul[0].header['CD2_2'] * 3600.0
    if (simulation and ares):
        hdul = pyfits.open(filealpha+'.fits')
        angx = hdul[0].data
        angy = hdul[1].data
        zl = hdul[0].header['ZLENS']
        Om0 = hdul[0].header['OMEGA']
        h = hdul[0].header['H']
        pix = hdul[0].header['CDELT2'] * 3600.0
        co = FlatLambdaCDM(Om0=Om0, H0=h*100.0)
        dls = co.angular_diameter_distance_z1z2(zl, zs).value
        ds = co.angular_diameter_distance(zs).value
        angx = angx * dls/ds
        angy = angy * dls/ds
    else:
        hdul = pyfits.open(filealpha + '_angx.fits')
        angx = hdul[0].data
        hdul = pyfits.open(filealpha + '_angy.fits')
        angy = hdul[0].data
        pix = hdul[0].header['CDELT2'] * 3600.0
    return angx, angy, zl, zs, pix


def sampleLensingDistance(co, zl, zsmax=6.0, nplanes=20):
    dl = co.angular_diameter_distance(zl)
    z = np.linspace(zl + 0.1, zsmax, 100)
    dls = co.angular_diameter_distance_z1z2(zl, z)
    ds = co.angular_diameter_distance(z)
    dlens = (dl * dls / ds).value
    f = interp1d(dlens, z)
    dlens_ = np.linspace(dlens.min(), dlens.max(), nplanes)
    z_ = f(dlens_)
    return z_


def placeSource(ys1, ys2, ycau1, ycau2, zs, size, npix, df, co, xc, yc):
    ds = co.angular_diameter_distance(zs).value
    # generate 6 random numbers from a uniform distribution (between 0 and 1)
    rn = np.random.rand(6)
    # axis ratio
    f = 0.95  # the axis ratio will vary between (1-f) and 1
    q = rn[0] * f + (1.0 - f)
    # effective radius
    rmax = 10.0  # kpc, the maximum physical size (re) of the sources
    rmin = 0.1  # kpc, the minimum physical size (re) of the sources
    rk = rn[1] * (rmax - rmin) + rmin
    re = np.rad2deg(rk / 1000.0 / ds) * 3600.0  # convert re from kpc to arcsec
    # position angle
    pa = rn[2] * np.pi
    # sersic index
    nmin = 0.5
    nmax = 4.0
    n = rn[3] * (nmax - nmin) + nmin
    # source position
    d = np.sqrt((ys1 - ycau1) ** 2 + (ys2 - ycau2) ** 2) + 0.2 * re
    dd = rn[4] * d
    phi = np.arctan2(ys2 - ycau2, ys1 - ycau1)
    dy1 = dd * np.cos(phi)  # (rn[4]-0.5)*re# allow a shift by up to half the effective radius
    dy2 = dd * np.sin(phi)  # (rn[5]-0.5)*re

    # set up source
    kwargs = {
        'n': n,
        'q': q,
        'ys1': ycau1 + dy1,  # ys1+dy1,
        'ys2': ycau2 + dy2,  # ys2+dy2,
        'pa': pa,
        're': re,
        'flux': 1.0,
        'zs': zs
    }
    # print (ycau1,ycau2,d,ys1,ys2,np.rad2deg(phi))
    sizex = [xc - size / 2.0, xc + size / 2.0]
    sizey = [yc - size / 2.0, yc + size / 2.0]
    se = sersic(size=size, Npix=npix, gl=df, save_unlensed=True,
                rmaxf=10.0, sizex=sizex, sizey=sizey, **kwargs)
    return se


def getSed(sedmod, rsed=None, zs=0.0, imin=0, imax=None):
    """
    Generate a SED. If ised is provided, then use the ised SED from the sedmod list. Otherwise,
    randomly generate SED between sed templates imin and imax
    :param sedmod: sed model
    :param rsed: float specifying the SED to be returned (optional)
    :param zs: source redshift
    :param imin: minimum index of SED templates to be used
    :param imax: maximum index of SED templates to be used
    :return: SED template
    """
    if rsed != None:
        sed = sedmod.getrSED(rsed=rsed, z=zs)
    else:
        if imax == None or imax > len(sedmod.seds_df.sed) - 2:
            imax = len(sedmod.seds_df.sed) - 2
        rn = np.random.rand(1) * (imax - imin + 1) + imin
        sed = sedmod.getrSED(rsed=rn, z=zs)
    return sed


def getMagFlux(sed, filterin, magin, filterout):
    """
    Given a SED normalized to a given magnitude magin in the filter filterin, returns
    magnitude and flux in the target filter filterout
    :param sed: source SED
    :param filterin: normalizing filter name
    :param magin: normalizing source magnitude in filterin
    :param filterout: filter for which the output magnitude is computed
    :return: magitude and flux in filter filterout
    """
    passband = ss.Passband(filterin)
    sed.normalise_to_mag(magin, passband)
    passband_ = ss.Passband(filterout)
    flux_ = sed.calc_flux(passband_)
    mag_ = sed.calc_mag(passband_, system='AB')
    flux_nu_ = ss.flambda_to_fnu(passband_.effective_wa, flux_)
    return mag_, flux_nu_


def formatHeader(hdu, pix, npix, zl, zs, crval1=0.0, crval2=0.0):
    hdu.header['RADESYS'] = 'ICRS    '
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CUNIT1'] = 'deg   '
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CUNIT2'] = 'deg   '

    hdu.header['CD1_1'] = -pix / 3600.0
    hdu.header['CD1_2'] = 0.000000000000E+00
    hdu.header['CD2_1'] = 0.000000000000E+00
    hdu.header['CD2_2'] = pix / 3600.0

    hdu.header['NAXIS'] = 2
    hdu.header['CRPIX1'] = npix // 2
    hdu.header['CRPIX2'] = npix // 2
    hdu.header['CRVAL1'] = crval1
    hdu.header['CRVAL2'] = crval2
    hdu.header['NAXIS1'] = npix
    hdu.header['NAXIS2'] = npix
    hdu.header['ZL'] = zl
    hdu.header['ZS'] = zs


def detect_arcs(image, threshf=0.01, min_area=5):
    # apply threshold
    thresh = threshf  # image.max()*threshf
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    minr = []
    minc = []
    maxr = []
    maxc = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= min_area:
            # draw rectangle
            minr_, minc_, maxr_, maxc_ = region.bbox
            minr.append(minr_)
            minc.append(minc_)
            maxr.append(maxr_)
            maxc.append(maxc_)

    minr = np.array(minr)
    maxr = np.array(maxr)
    minc = np.array(minc)
    maxc = np.array(maxc)
    return image_label_overlay, minr, minc, maxr, maxc


def toas(x, fov=100.0, npix=2048):
    y = (x - npix / 2.0) * fov / (npix - 1)
    return y


def plot_regions(minr, minc, maxr, maxc, delta=0.0, ax=None):
    for i in range(len(minr)):
        rect = mpatches.Rectangle((minc[i] - delta / 2, minr[i] - delta / 2), maxc[i] - minc[i] + delta,
                                  maxr[i] - minr[i] + delta,
                                  fill=False, edgecolor='blue', linewidth=1)
        ax.add_patch(rect)


def arcsec2ra(x, y, RA_REF, DEC_REF):
    RA = -x / (np.cos(np.deg2rad(DEC_REF)) * 3600.0) + RA_REF
    DEC = y / 3600.0 + DEC_REF
    return RA, DEC


from matplotlib.backends.backend_pdf import PdfPages

pdf_pages = PdfPages('ggsl_examples.pdf')
for cli in range(nclus):
    clnum = '%0*d' % (4, cli + 1)
    # process each projection of cluster cli
    for pr in range(npr):
        if (simulation and not ares):
            filealpha = path_to_clusters + 'angles_' + clnum + '_' + snap + '_' + str(pr) + '.fits'
        elif (simulation and ares):
            filealpha = path_to_clusters + cln
        else:
            filealpha = path_to_clusters + cln
        # read cluster deflection angle
        angx, angy, zl, zs, pix = readClusterDA(filealpha, simulation=simulation, ares=ares, zl=zl, zs=zs)
        # create an array of source reshifts that sample the lensing distance curve
        zs_arr = sampleLensingDistance(co, zl, nplanes=nzs)
        # build deflector
        kwargs_def = {'zl': zl, 'zs': zs}
        df = deflector(co=co, angx=angx, angy=angy, **kwargs_def)
        theta = np.linspace(-angx.shape[0] * pix / 2.0, angx.shape[0] * pix / 2.0, angx.shape[0])
        df.setGrid(theta=theta, compute_potential=False)
        # prepare output file:
        if simulation:
            file_out = 'ggsl_' + clnum + '_' + snap + '_' + str(pr) + '.fits'
            cat_out = 'ggsl' + clnum + '_' + snap + '_' + str(pr) + '.pkl'
        else:
            file_out = 'ggsl_' + cln + '.fits'
            cat_out = 'ggsl_' + cln + '.pkl'
        initFits = True
        hdul_out = []
        # loop on source planes
        arcs = []
        for zs_ in tqdm(zs_arr):
            initcau = False
            initcrit = False
            df.change_redshift(zs_)
            # simulate tangential arcs: find tangential critical lines
            tcl = df.tancl()
            # also find radial critical lines and caustics
            rcl = df.radcl()
            if rcl.size > 0:
                rca = df.getCaustics(rcl)
            # if any tangential critical line exist, start processing
            if tcl.size > 0:
                # map critical lines into caustics
                tca = df.getCaustics(tcl)
                # for each caustic, collect point coordinates and pack them into two arrays
                # only use points on the secondary critical lines

                for jj in range(len(tcl)):
                    if (tcl[jj].getThetaE() * pix < maxthetaE) & (tcl[jj].getThetaE() * pix > minthetaE) & (
                    not tcl[jj].principale):
                        x_, y_ = df.getCausticPoints(tca[jj])
                        xca = x_.mean()
                        yca = y_.mean()
                        if not initcau:
                            x = x_
                            y = y_
                            xcau = np.ones_like(x_) * xca
                            ycau = np.ones_like(y_) * yca
                            initcau = True
                        else:
                            x = np.concatenate((x, x_))
                            y = np.concatenate((y, y_))
                            xcau__ = np.ones_like(x_) * xca
                            ycau__ = np.ones_like(y_) * yca
                            xcau = np.concatenate((xcau, xcau__))
                            ycau = np.concatenate((ycau, ycau__))

                        xc_, yc_ = df.getCritPoints(tcl[jj])
                        xm = xc_.mean()  # tcl[jj].px
                        ym = yc_.mean()  # tcl[jj].py
                        if not initcrit:
                            xcrit = np.ones_like(xc_) * xm
                            ycrit = np.ones_like(yc_) * ym
                            initcrit = True
                        else:
                            xcrit__ = np.ones_like(xc_) * xm
                            ycrit__ = np.ones_like(yc_) * ym
                            xcrit = np.concatenate((xcrit, xcrit__))
                            ycrit = np.concatenate((ycrit, ycrit__))

                # randomly select narcs caustic points
                ri = np.random.randint(0, x.size, size=narcs)

                # place sources at the selected caustic points (with shifts allowed)
                # pack images into a fits file
                for i in range(narcs):

                    src = placeSource(x[ri[i]], y[ri[i]], xcau[ri[i]], ycau[ri[i]], zs=zs_, size=fov, npix=npix, df=df,
                                      co=co,
                                      xc=xcrit[ri[i]], yc=ycrit[ri[i]])
                    ra_c, dec_c = arcsec2ra(xcrit[ri[i]], ycrit[ri[i]], RA_REF, DEC_REF)
                    # assign a sed to the source
                    sed = getSed(sedm, zs=zs_, imin=5, imax=10)
                    mag__, flux__ = getMagFlux(sed, filterin=filtern, magin=magn, filterout=filtern)
                    # compute flux ratios between bands
                    flr = []
                    flux = []
                    mag = []
                    for j in range(len(filtern_FF)):
                        mag_, flux_ = getMagFlux(sed, filterin=filtern, magin=magn, filterout=filtern_FF[j])
                        flr.append(flux_ / flux__)
                        flux.append(flux_)
                        mag.append(mag_)

                    image_label_overlay, minr, minc, maxr, maxc = detect_arcs(src.image, threshf=0.001)
                    arcs.append({'minr': minr, 'minc': minc, 'maxr': maxr, 'maxc': maxc, 'flr': flr, 'sed': sed, 'se': src})
                    minr = toas(minr, fov=fov, npix=npix)
                    minc = toas(minc, fov=fov, npix=npix)
                    maxr = toas(maxr, fov=fov, npix=npix)
                    maxc = toas(maxc, fov=fov, npix=npix)

                    if initFits:
                        primary_hdu = pyfits.PrimaryHDU(src.image)
                        formatHeader(primary_hdu, fov / npix, npix, zl, zs_, crval1=ra_c, crval2=dec_c)
                        hdul_out.append(primary_hdu)
                        initFits = False
                    else:
                        hdu = pyfits.ImageHDU(src.image)
                        formatHeader(hdu, fov / npix, npix, zl, zs_, crval1=ra_c, crval2=dec_c)
                        hdul_out.append(hdu)
                    if showexamples:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                        #############
                        ax.imshow(np.sqrt(src.image), origin='lower', cmap='cubehelix_r',
                                  extent=[-fov / 2., fov / 2., -fov / 2., fov / 2.])

                        # plot_regions(minr,minc,maxr,maxc,delta=1.0,ax=ax)

                        for crit in tcl:
                            xc_, yc_ = df.getCritPoints(crit)
                            ax.plot(xc_ - xcrit[ri[i]], yc_ - ycrit[ri[i]], '-', color='red', lw=1, alpha=0.5)
                        for crit in rcl:
                            xc_, yc_ = df.getCritPoints(crit)
                            ax.plot(xc_ - xcrit[ri[i]], yc_ - ycrit[ri[i]], '-', color='red', lw=1, alpha=0.5)
                        ax.set_xlim([-fov / 2., fov / 2.])
                        ax.set_ylim([-fov / 2., fov / 2.])
                        pdf_pages.savefig(fig, dpi=150)
                        plt.close(fig)
        hdul = pyfits.HDUList(hdul_out)
        hdul.writeto(file_out, overwrite=True)
        dff = pd.DataFrame(arcs)
        with open(cat_out, "wb") as f:
            pickle.dump(dff, f)
pdf_pages.close()
