from pyLensLib.sersic import sersic
from pyLensLib import samplers as sa
import numpy as np
import matplotlib.pyplot as plt
import pyLensLib.lenstool as lst
# these imports are needed to work with SEDs
from pyLensLib.sedmodel import sedmodel
import pyLensLib.sedcompat as ss
from tqdm import tqdm

# these imports are needed to simulate observations
from pyLensLib.observation import observation

from icecream import ic
np.random.seed(10)

############ SED models and filters

sedm = sedmodel(sed_dir='/Users/massimo/CODES/bpz-1.99.3/SED/',sed_list='eB11.list')

filtern_out = ['BPZ/HST_ACS_WFC_F435W.res',
              'BPZ/HST_ACS_WFC_F606W.res',
              'BPZ/HST_ACS_WFC_F814W.res']

filtern = 'BPZ/HST_ACS_WFC_F814W.res'

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
    :return: magnitude and flux in filter filterout
    """
    passband = ss.Passband(filterin)
    sed.normalise_to_mag(magin, passband)
    passband_ = ss.Passband(filterout)
    mag_ = sed.calc_mag(passband_, system='AB')

    return mag_

############ Interface to observed cluster database

cluster_rgb = ['/Users/massimo/stiva/MACS1206/CLASH/macs1206_RGB.fits',
               '/Users/massimo/stiva/RGBs/macs0416_ff_30mas_RGB.fits',
               '/Users/massimo/stiva/RGBs/2248_ff_RGB.fits',
               '/Users/massimo/stiva/clusters/PSZ1G311/hst_images/j155004m7811-30mas-ir_drz_sci.fits'
               '/Users/massimo/stiva/RGBs/A370_RGB.fits',
               '/Users/massimo/stiva/RGBs/abell2744_RGB.fits',
               '/Users/massimo/stiva/RGBs/macs0717_RGB.fits',
               '/Users/massimo/stiva/RGBs/1149_60mas_ff.fits_RGB.fits',
               '/Users/massimo/stiva/RGBs/macs0329_RGB.fits',
               '/Users/massimo/stiva/RGBs/macs1931_RGB.fits',
               '/Users/massimo/stiva/RGBs/macs2129_RGB.fits',
               '/Users/massimo/stiva/RGBs/rxj2129_RGB.fits']

# cluster nickname (necessari per costruire i nomi dei file da leggere)
cluster = ['S1063','M0416','M1206pl','PSZ1G311_200',
           'A370','A2744','M0717','M1149',
           'M0329','M1931','M2129','R2129']

# posizione delle mappe degli angoli di deflessione
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


# redshift delle lenti
#zl_ = [0.439,0.396,0.348,0.4436,0.375,0.308,0.545,0.543,0.45,0.352,0.587,0.234]

icl = 2
zs = 1.03

zl_s = lst.getLensRedshift(path_to_angles[icl] + cluster[icl] + '.par')
print (('Working with cluster %s at redshift %s') % (cluster[icl], zl_s))
zl = float(zl_s)

print ('creating deflector...')
df = lst.create_deflector(parfile = path_to_angles[icl] + cluster[icl] + '.par',
                          filex = path_to_angles[icl] + cluster[icl] + '_angx.fits',
                          filey = path_to_angles[icl] + cluster[icl] + '_angy.fits',
                          zl= zl, zs = zs, zsnorm = 1.0, resc_fact = 1.0)

print ('cosmological parameters: H0=%s, Om0=%s, OmL-%s' % (df.co.H0, df.co.Om0, df.co.Ode0))
co = df.co

sx,sy,fovsp = df.fovSP()

# set up source and observe it
magn = 23.0
npix = 1000 #df.a1.shape[0]#500
fsize = 50.0 #np.max(df.thetax)-np.min(df.thetax)
pixel = fsize / (npix - 1)

mlim = 30.0
sn = 10
rap = 1.0
ob = observation(size=fsize, Npix=npix, texp=10000.0, zp=23.9, rap=rap, mlim=mlim, sn=sn)
print (('############################### \n'
       'Instanceating an observation... \n'
       '- assuming mlim=%3.1f \n'
        '- required signal-to-noise %3.1f') % (mlim,sn))

fsub = 0.05 #0.1
sizex = [-fsize/2.0, fsize/2.0]
sizey = [-fsize/2.0, fsize/2.0]

# Assign a SED to the host galaxy: here we assume the host has a SED typycal of a spiral galaxy
sed = getSed(sedm,zs=zs,imin=0,imax=2)
#sed = getSed(sedm,zs=zs,imin=4,imax=6)
#sed = getSed(sedm,zs=zs,imin=6,imax=8)

# Generate source and clumps in a reference band
flgal = ob.mag2counts(magn)
print (('############################### \n'
        'Creating the host galaxy: \n'
       '- the reference band is %s \n'
       '- the imput magnitude is %5.2f AB mag \n'
       '- the resulting flux is %8.4f c/s \n')
       % (filtern, magn, flgal))

# compute source position based on the observed image position

kwargs = {
    'n': 1.0,
    'q': 0.3,
    'ys1': 4.8798,#28.3951950,  # 5.5,  # ys1+dy1,
    'ys2': 2.0026,#40.2729467,  # 2.1,  # ys2+dy2,
    'pa': np.pi / 4.0,
    're': 1.0,
    'flux': (1.0 - fsub) * flgal,
    'zs': zs
}

BT = 0.5

kwargs = {
    'n': 1.0,
    'q': 0.5,
    'ys1': 5.099,#28.3951950,  # 5.5,  # ys1+dy1,
    'ys2': 2.1588,#40.2729467,  # 2.1,  # ys2+dy2,
    'pa': np.pi / 8.0,
    're': 0.3,
    'flux': (1.0 - fsub - BT) * flgal,
    'zs': zs
}

#####
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score, mean_squared_error

import warnings
warnings.filterwarnings('ignore')
# Create a model
model = RANSACRegressor(base_estimator=LinearRegression(),
						min_samples=50, max_trials=100,
						loss='absolute_error', random_state=42,
						residual_threshold=10)

# Fit the model
data = pd.read_csv('../data/BTn.csv',header=None,sep=',')
data.columns=['n','bt']
y = data.n.to_numpy().reshape(-1, 1)
X = data.bt.to_numpy().reshape(-1, 1)
model.fit(X, y)
print('Slope:%.3f;Intercept:%.3f'%(model.estimator_.coef_[0],model.estimator_.intercept_))


Y = model.predict(X)
plt.plot(X,y,'o')
plt.plot(X,Y,'-')
plt.show()


lognb = 0.4 * np.log(max(BT,0.03))#+0.1*(np.random.rand(1)*2-1.0)
nb = np.exp(lognb)
ic (nb)
ic (5.0/np.log10(0.5))

kwargs_bulge = {
    'n': 4,
    'q': 0.9,
    'ys1': 5.099,#28.3951950,  # 5.5,  # ys1+dy1,
    'ys2': 2.1588,#40.2729467,  # 2.1,  # ys2+dy2,
    'pa': np.pi / 8.0,
    're': 0.3,
    'flux': flgal*BT,
    'zs': zs
}

sizex=[ 20.6271 - 25, 20.6271+15]
sizey=[-2.2179 - 10, -2.2179+30]

#sizex = [-fsize/2.0, fsize/2.0]
#sizey = [-fsize/2.0, fsize/2.0]

size_unlensed = 5.0
npix_unlensed = npix
pixel_unlensed = size_unlensed/(npix_unlensed - 1)

se = sersic(sizex=sizex, sizey=sizey,Npix=npix, gl=df, save_unlensed=True,
            npix_unlensed=npix_unlensed, save_unlensed_recenter=True, size_unlensed=size_unlensed, y1_unlensed=kwargs['ys1'],
            y2_unlensed=kwargs['ys2'], rmaxf=10.0, **kwargs)
vmax = np.sqrt(se.image_unlensed).max()

se_bulge = sersic(sizex=sizex, sizey=sizey,Npix=npix, gl=df, save_unlensed=True,
            npix_unlensed=npix_unlensed, save_unlensed_recenter=True, size_unlensed=size_unlensed, y1_unlensed=kwargs['ys1'],
            y2_unlensed=kwargs['ys2'], rmaxf=10.0, **kwargs_bulge)

fig,ax =plt.subplots(2,1,figsize=(10,18))
ax[0].imshow(np.sqrt(se.image_unlensed),cmap='cubehelix',origin='lower',extent=[-2.5,2.5,-2.5,2.5],vmax=vmax)
ax[1].imshow(np.sqrt(se.image_unlensed+se_bulge.image_unlensed),cmap='cubehelix',origin='lower',extent=[-2.5,2.5,-2.5,2.5],vmax=vmax)
for i in range(2):
    ax[i].set_xlabel(r'$\theta_1$ [arcsec]',fontsize=20)
    ax[i].set_ylabel(r'$\theta_2$ [arcsec]',fontsize=20)
    ax[i].xaxis.set_tick_params(labelsize=20)
    ax[i].yaxis.set_tick_params(labelsize=20)
#plt.tight_layout()
fig.savefig('host_example.png',bbox_inches='tight',dpi=300)

l = np.linspace(0.001, 1.0, 10000)

l_ = sa.samplePowerLawExCut(xmin=0.0001,xcut=0.01,fsub=fsub,delta=1.0,beta=-2.0,gamma=3.0)
ic(np.sum(l_))

l_ = np.array(l_)
isel = l_> 0.0001
l = l_[isel]

#ic(l)


image=se.image_unlensed.copy()
image[image<(image.max()*0.05)]=0.0

print (image.min(),image.max())

x,y = sa.sample2Dimage(image.T,n=len(l))
x_arcsec, y_arcsec = (x-npix_unlensed/2.0+0.5) * pixel_unlensed + kwargs['ys1'], \
                     (y-npix_unlensed/2.0+0.5) * pixel_unlensed + kwargs['ys2']

fig,ax =plt.subplots(1,1,figsize=(10,10))
ax.imshow(np.sqrt(se.image_unlensed),cmap='cubehelix',origin='lower',extent=[-2.5,2.5,-2.5,2.5],vmax=vmax)

ax.set_xlabel(r'$\theta_1$ [arcsec]',fontsize=20)
ax.set_ylabel(r'$\theta_2$ [arcsec]',fontsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.plot(x_arcsec,y_arcsec,'o',color='red')
#plt.tight_layout()
fig.savefig('host_clumps_example.png',bbox_inches='tight',dpi=300)


# now we can assign a SED to each clump. We assume that the clumps are active star forming regions
# we use star burst templates:
sedc = []
for i in range(len(x_arcsec)):
    sed__ = getSed(sedm, zs=zs, imin=5, imax=9) # 7, 9
    sedc.append(sed__)

# compute host and clump flux in additional bands
ic(len(filtern_out),len(x_arcsec))
fluxes_host_norm = np.zeros(len(filtern_out))
fluxes_clumps_norm = np.zeros((len(filtern_out),len(x_arcsec)))

print ("Computing fluxes in different bands")
for i in range(len(filtern_out)):
    mag__ = getMagFlux(sed,filterin=filtern,magin=magn,filterout=filtern_out[i])
    fluxes_host_norm[i] = 10**(-0.4*(mag__-magn))
    for j in tqdm(range(len(x_arcsec))):
        mag__ = getMagFlux(sedc[j], filterin=filtern, magin=magn, filterout=filtern_out[i])
        fluxes_clumps_norm[i,j] = 10**(-0.4*(mag__-magn))

plotThisFigure=False
if plotThisFigure:
    fig,ax=plt.subplots(1,1,figsize=(20,20))
    ax.imshow(se.image_unlensed,origin='lower',extent=[-fsize/2.,fsize/2.,-fsize/2.,fsize/2.])
    ax.plot(x_arcsec,y_arcsec,'o',color='red',alpha=0.8)
    ax.plot(se.ys1,se.ys2,'o',color='yellow')
    plt.show()

cube_image = np.zeros((npix,npix,3))
cube_limage = np.zeros((npix,npix,3))
print ("Painting source.")
for j in range(len(filtern_out)):
    image = se.image_unlensed.copy() * fluxes_host_norm[j]
    limage = se.image.copy() * fluxes_host_norm[j]

    lsub = l * flgal * fluxes_clumps_norm[j,:]
    for i in tqdm(range(len(l))):
        kwargs_sub = {
            'n': 4.0,
            'q': 1.0,
            'ys1': x_arcsec[i],  # ys1+dy1,
            'ys2': y_arcsec[i],  # ys2+dy2,
            'pa': 0.0,
            're': 0.1*(l[i]/(0.01))**(1./3.),
            'flux': lsub[i],
            'zs': zs
        }
        se_sub = sersic(sizex=sizex, sizey=sizey, Npix=npix, gl=df, save_unlensed=True, save_unlensed_recenter=True,
                        npix_unlensed=npix, size_unlensed=5.0, y1_unlensed=kwargs['ys1'], y2_unlensed=kwargs['ys2'],
                        rmaxf=10.0, **kwargs_sub)
        image= image + se_sub.image_unlensed.copy()
        limage = limage + se_sub.image.copy()
    cube_image[:,:,j] = image
    cube_limage[:,:,j] = limage





tl=df.tancl() #critica tang
rl=df.radcl() #critica rad
if rl.size > 0:
    rcl=df.getCaustics(rl)
if tl.size > 0:
    tcl=df.getCaustics(tl)

fig,ax=plt.subplots(1,1,figsize=(20,20))
ax.imshow(cube_image,origin='lower',
          extent=[-fsize/2.,fsize/2.,-fsize/2.,fsize/2.])
#for t in tcl:
#    u,v = df.getCausticPoints(t)
#    ax.plot(u,v,'-',color='yellow')

#for r in rcl:
#    m,n = df.getCausticPoints(r)
#    ax.plot(m,n,'-',color='yellow')

ax.set_xlim([-fsize/2.,fsize/2.])
ax.set_ylim([-fsize/2.,fsize/2.])

ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.set_aspect('equal')

fig.savefig('clumpygal.pdf',bbox_inches='tight',dpi=300)


fig,ax=plt.subplots(1,1,figsize=(20,20))
ax.imshow(cube_limage,origin='lower',
          extent=[-fsize/2.,fsize/2.,-fsize/2.,fsize/2.])
#ax.plot(x,y,'o',color='red',alpha=0.8)

for t in tl:
    u,v = df.getCritPoints(t)
    ax.plot(u,v,'-',color='yellow')

for r in rl:
    m,n = df.getCritPoints(r)
    ax.plot(m,n,'-',color='yellow')


ax.set_xlim([-fsize/2.,fsize/2.])
ax.set_ylim([-fsize/2.,fsize/2.])

ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.set_aspect('equal')
#plt.show()

fig.savefig('lensedclumpygal.pdf',bbox_inches='tight',dpi=300)


def formatHeader(hdu,pix,npix,zl,zs):
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['RADESYS'] = 'FK5    '
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CUNIT1'] = 'deg   '
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CUNIT2'] = 'deg   '

    hdu.header['CD1_1'] = -pix/3600.0
    hdu.header['CD1_2'] = 0.000000000000E+00
    hdu.header['CD2_1'] = 0.000000000000E+00
    hdu.header['CD2_2'] = pix/3600.0

    hdu.header['NAXIS'] = 2
    hdu.header['CRPIX1'] = npix//2 + 0.5
    hdu.header['CRPIX2'] = npix//2 + 0.5
    hdu.header['CRVAL1'] = 0.0
    hdu.header['CRVAL2'] = 0.0
    hdu.header['NAXIS1'] = npix
    hdu.header['NAXIS2'] = npix
    hdu.header['ZL'] = zl
    hdu.header['ZS'] = zs


import astropy.io.fits as pyfits



initFits = True
fileout = ['b.fits','g.fits','r.fits']
hdul_out=[]
for i in range(3):
    primary_hdu = pyfits.PrimaryHDU(cube_image[:,:,i])
    formatHeader(primary_hdu, fsize / (npix-1), npix, 0.0, zs)
    hdul = pyfits.HDUList([primary_hdu])
    hdul.writeto(fileout[i], overwrite=True)


fileout = ['b_lensed.fits','g_lensed.fits','r_lensed.fits']
hdul_out=[]
for i in range(3):
    primary_hdu = pyfits.PrimaryHDU(cube_limage[:,:,i])
    print (np.min(cube_limage[:,:,i]),np.max(cube_limage[:,:,i]))
    formatHeader(primary_hdu, fsize / (npix-1), npix, 0.0, zs)
    hdul = pyfits.HDUList([primary_hdu])
    hdul.writeto(fileout[i], overwrite=True)
