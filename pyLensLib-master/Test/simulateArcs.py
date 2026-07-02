import numpy as np
from pyLensLib.deflector import deflector
from pyLensLib.sersic import sersic
import matplotlib.pyplot as plt
from tqdm import tqdm
import astropy.io.fits as pyfits
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from pyLensLib.maputils import map_obj
from pyLensLib.sedmodel import sedmodel
import pyLensLib.sedcompat as ss
import pickle
import pandas as pd

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches

from acstools import acszpt

sedm = sedmodel(sed_dir='/Users/massimo/CODES/bpz-1.99.3/SED/',sed_list='eB11.list')

filtern_FF = ['BPZ/HST_ACS_WFC_F435W.res',
              'BPZ/HST_ACS_WFC_F814W.res',
              'BPZ/HST_WFC3_IR_F105W.res',
              'BPZ/HST_WFC3_IR_F125W.res',
              'BPZ/HST_WFC3_IR_F140W.res',
              'BPZ/HST_WFC3_IR_F160W.res']

filtern = 'BPZ/HST_ACS_WFC_F814W.res'
magn = 20.0

simulation=False

if simulation:
    # assume to process several simulated clusters
    nclus = 1
    snap = '115'
    path_to_clusters = '/Users/massimo/stiva/300/SL/'
    npr = 3
    zl = None
    zs = None
else:
    # process a real cluster
    nclus = 1
    snap = ''
    path_to_clusters = '/Users/massimo/stiva/pietro_models/'
    cln='M0416'
    npr = 1
    zl = 0.396
    zs = 1.0

# show examples?
showexamples=True

# set cosmology
co = FlatLambdaCDM(Om0=0.3,H0=70.0)

# number of arcs simulated for each iteration
narcs=20
nzs=10

# images
fov=150.0
npix=1024

def readClusterDA(filealpha,simulation=True,zl=None,zs=None):
    if simulation:
        hdul = pyfits.open(filealpha)
        angx = hdul[0].data
        angy = hdul[1].data
        zl = hdul[0].header['ZL']
        zs = hdul[0].header['ZS']
        pix = hdul[0].header['CD2_2']*3600.0
    else:
        hdul = pyfits.open(filealpha+'_angx.fits')
        angx = hdul[0].data
        hdul = pyfits.open(filealpha+'_angy.fits')
        angy = hdul[0].data
        pix = hdul[0].header['CDELT2'] * 3600.0
    return angx, angy, zl, zs, pix

def sampleLensingDistance(co,zl,zsmax=6.0,nplanes=20):
    dl = co.angular_diameter_distance(zl)
    z=np.linspace(zl+0.1,zsmax,100)
    dls = co.angular_diameter_distance_z1z2(zl,z)
    ds = co.angular_diameter_distance(z)
    dlens = (dl*dls/ds).value
    f = interp1d(dlens, z)
    dlens_ = np.linspace(dlens.min(),dlens.max(),nplanes)
    z_ = f(dlens_)
    return z_

def placeSource(ys1,ys2,zs,size,npix,df,co):
    ds=co.angular_diameter_distance(zs).value
    # generate 6 random numbers from a uniform distribution (between 0 and 1)
    rn=np.random.rand(6)
    # axis ratio
    f=0.95 # the axis ratio will vary between (1-f) and 1
    q=rn[0]*f+(1.0-f)
    # effective radius
    rmax=10.0 # kpc, the maximum physical size (re) of the sources
    rmin=0.1 # kpc, the minimum physical size (re) of the sources
    rk=rn[1]*(rmax-rmin)+rmin
    re=np.rad2deg(rk/1000.0/ds)*3600.0 # convert re from kpc to arcsec
    # position angle
    pa=rn[2]*np.pi
    # sersic index
    nmin=0.5
    nmax=4.0
    n=rn[3]*(nmax-nmin)+nmin
    # source position
    dy1 = (rn[4]-0.5)*re# allow a shift by up to half the effective radius
    dy2 = (rn[5]-0.5)*re
    # set up source
    kwargs = {
        'n': n,
        'q': q,
        'ys1': ys1+dy1,
        'ys2': ys2+dy2,
        'pa': pa,
        're': re,
        'flux': 1.0,
        'zs': zs
    }
    #print (kwargs)
    se = sersic(size=size, Npix=npix, gl=df, save_unlensed=True, rmaxf=10.0, **kwargs)
    return se

def getSed(sedmod,rsed=None,zs=0.0,imin=0,imax=None):
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
        sed = sedmod.getrSED(rsed=rsed,z=zs)
    else:
        if imax == None or imax > len(sedmod.seds_df.sed)-2:
            imax = len(sedmod.seds_df.sed)-2
        rn = np.random.rand(1)*(imax-imin+1)+imin
        sed = sedmod.getrSED(rsed=rn,z=zs)
    return sed

def getMagFlux(sed,filterin,magin,filterout):
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
    flux_nu_=ss.flambda_to_fnu(passband_.effective_wa, flux_)
    return mag_,flux_nu_

def formatHeader(hdu,pix,npix,zl,zs):
    hdu.header['RADESYS'] = 'ICRS    '
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CUNIT1'] = 'deg   '
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CUNIT2'] = 'deg   '

    hdu.header['CD1_1'] = -pix/3600.0
    hdu.header['CD1_2'] = 0.000000000000E+00
    hdu.header['CD2_1'] = 0.000000000000E+00
    hdu.header['CD2_2'] = pix/3600.0

    hdu.header['NAXIS'] = 2
    hdu.header['CRPIX1'] = npix//2
    hdu.header['CRPIX2'] = npix//2
    hdu.header['CRVAL1'] = 0.0
    hdu.header['CRVAL2'] = 0.0
    hdu.header['NAXIS1'] = npix
    hdu.header['NAXIS2'] = npix
    hdu.header['ZL'] = zl
    hdu.header['ZS'] = zs

def flux2counts(flux_hst, header_hst):
    '''
    Convert the flux F_nu into HST counts (e-/s)
    :param flux_hst: HST flux F_nu
    :param header_hst: HST header object
    :return: HST counts in e-/s
    '''

    abzp_hst = -2.5 * np.log10(header_hst['PHOTFLAM']) - 5 * np.log10(header_hst['PHOTPLAM']) - 2.4079

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

    cost = (abzp_hst + 48.60) / 2.5
    counts_hst = flux_hst * 10**cost

    return counts_hst

def detect_arcs(image,threshf=0.01,min_area=5):
    # apply threshold
    thresh = threshf#image.max()*threshf
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    minr=[]
    minc=[]
    maxr=[]
    maxc=[]
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

def toas(x,fov=100,npix=2048):
    y=(x-npix/2.0)*fov/(npix-1)
    return y

def plot_regions(minr,minc,maxr,maxc,delta=0.0,ax=None):
    for i in range(len(minr)):
        rect = mpatches.Rectangle((minc[i]-delta/2, minr[i]-delta/2), maxc[i] - minc[i] + delta, maxr[i] - minr[i] + delta,
                                  fill=False, edgecolor='blue', linewidth=1)
        ax.add_patch(rect)

from matplotlib.backends.backend_pdf import PdfPages
pdf_pages = PdfPages('arc_examples.pdf')
for cli in range(nclus):
    clnum = '%0*d' % (4, cli + 1)
    # process each projection of cluster cli
    for pr in range(npr):
        if simulation:
            filealpha = path_to_clusters + 'angles_'+clnum+'_'+snap+'_'+str(pr)+'.fits'
        else:
            filealpha = path_to_clusters + cln
        # read cluster deflection angle
        angx,angy,zl,zs,pix = readClusterDA(filealpha,simulation=simulation,zl=zl,zs=zs)
        # create an array of source reshifts that sample the lensing distance curve
        zs_arr = sampleLensingDistance(co,zl,nplanes=nzs)
        # build deflector
        kwargs_def = {'zl': zl, 'zs': zs}
        df = deflector(co=co,angx=angx,angy=angy,**kwargs_def)
        theta = np.linspace(-angx.shape[0]*pix/2.0,angx.shape[0]*pix/2.0,angx.shape[0])
        df.setGrid(theta=theta,compute_potential=False)
        # prepare output file:
        if simulation:
            file_out = 'arcs_' + clnum + '_' + snap + '_' + str(pr) + '.fits'
            cat_out = 'arcs' + clnum + '_' + snap + '_' + str(pr) + '.pkl'
        else:
            file_out = 'arcs_' + cln + '.fits'
            cat_out = 'arcs_' + cln + '.pkl'
        initFits= True
        hdul_out = []
        # loop on source planes
        arcs = []
        for zs_ in tqdm(zs_arr):
            initcau = False
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
                for cau in tca:
                    x_, y_ = df.getCausticPoints(cau)
                    if not initcau:
                        x = x_
                        y = y_
                        initcau = True
                    else:
                        x=np.concatenate((x,x_))
                        y=np.concatenate((y,y_))

                # randomly select narcs caustic points
                ri = np.random.randint(0,x.size,size=narcs)

                # place sources at the selected caustic points (with shifts allowed)
                # pack images into a fits file
                for i in range(narcs):
                    src=placeSource(x[ri[i]],y[ri[i]],zs=zs_,size=fov,npix=npix,df=df,co=co)
                    # assign a sed to the source
                    sed = getSed(sedm,zs=zs_,imin=5,imax=10)
                    mag__, flux__ = getMagFlux(sed,filterin=filtern,magin=magn,filterout=filtern)
                    # compute flux ratios between bands
                    flr=[]
                    flux=[]
                    mag=[]
                    for j in range(len(filtern_FF)):
                        mag_, flux_ = getMagFlux(sed,filterin=filtern,magin=magn,filterout=filtern_FF[j])
                        flr.append(flux_/flux__)
                        flux.append(flux_)
                        mag.append(mag_)
                    #print (flr)
                    #print ('mags:',mag)

                    #mpo = map_obj(src.image)
                    #cont = mpo.get_contours(src.image.max()*0.001,fully_connected='high')
                    image_label_overlay, minr, minc, maxr, maxc = detect_arcs(src.image, threshf=0.001)
                    arcs.append({'minr': minr, 'minc': minc, 'maxr': maxr, 'maxc': maxc, 'flr': flr})
                    minr = toas(minr,fov=fov,npix=npix)
                    minc = toas(minc, fov=fov, npix=npix)
                    maxr = toas(maxr, fov=fov, npix=npix)
                    maxc = toas(maxc, fov=fov, npix=npix)

                    if initFits:
                        primary_hdu=pyfits.PrimaryHDU(src.image)
                        formatHeader(primary_hdu, fov / npix, npix, zl, zs_)
                        hdul_out.append(primary_hdu)
                        initFits=False
                    else:
                        hdu = pyfits.ImageHDU(src.image)
                        formatHeader(hdu, fov / npix, npix, zl, zs_)
                        hdul_out.append(hdu)
                    if showexamples:
                        fig,ax=plt.subplots(1,2,figsize=(16,9))
                        #############
                        ax[0].imshow(np.sqrt(src.image),origin='lower',cmap='cubehelix_r',
                                     extent=[-fov/2.,fov/2.,-fov/2.,fov/2.])
                        #for c in cont:
                        #    xc_, yc_ = zip(*c.points)
                        #    xc_ = (np.array(xc_) - npix / 2.0) * fov / (npix - 1)
                        #    yc_ = (np.array(yc_) - npix / 2.0) * fov / (npix - 1)
                        #    #ax[0].plot(xc_,yc_,'-',color='yellow')
                        #    xcen = xc_.mean()
                        #    ycen = yc_.mean()
                        #   xmin = xc_.min()-5.0
                        #    xmax = xc_.max()+5.0
                        #    ymin = yc_.min()-5.0
                        #    ymax = yc_.max()+5.0
                        #    ax[0].plot([xmin,xmax],[ymin,ymin],'-',color='blue')
                        #    ax[0].plot([xmax, xmax], [ymin, ymax], '-', color='blue')
                        #    ax[0].plot([xmax, xmin], [ymax, ymax], '-', color='blue')
                        #    ax[0].plot([xmin, xmin], [ymax, ymin], '-', color='blue')
                        plot_regions(minr,minc,maxr,maxc,delta=1.0,ax=ax[0])

                        for crit in tcl:
                            xc_, yc_ = df.getCritPoints(crit)
                            ax[0].plot(xc_,yc_,'-',color='red',lw=0.5,alpha=0.5)
                        for crit in rcl:
                            xc_, yc_ = df.getCritPoints(crit)
                            ax[0].plot(xc_,yc_,'-',color='red',lw=0.5,alpha=0.5)

                        ##############
                        ax[1].imshow(np.sqrt(src.image_unlensed),origin='lower',cmap='cubehelix_r',
                                     extent=[-fov/2.,fov/2.,-fov/2.,fov/2.])
                        for cau in tca:
                            xc_, yc_ = df.getCausticPoints(cau)
                            ax[1].plot(xc_,yc_,'-',color='red',lw=0.5,alpha=0.5)
                        for cau in rca:
                            xc_, yc_ = df.getCausticPoints(cau)
                            ax[1].plot(xc_,yc_,'-',color='red',lw=0.5,alpha=0.5)

                        for i in range(2):
                            ax[i].set_xlim([-fov/2.,fov/2.])
                            ax[i].set_ylim([-fov/2.,fov/2.])
                        pdf_pages.savefig(fig,dpi=150)
                        plt.close(fig)
        hdul = pyfits.HDUList(hdul_out)
        hdul.writeto(file_out, overwrite=True)
        dff = pd.DataFrame(arcs)
        with open(cat_out, "wb") as f:
            pickle.dump(dff, f)
pdf_pages.close()








