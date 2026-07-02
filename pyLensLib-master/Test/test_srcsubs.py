from pyLensLib.sersic import sersic
from pyLensLib import samplers as sa
import numpy as np
import matplotlib.pyplot as plt
import pyLensLib.lenstool as lst
# these imports are needed to work with SEDs
from pyLensLib.sedmodel import sedmodel
import pyLensLib.sedcompat as ss

# these imports are needed to simulate observations
from pyLensLib.observation import observation

from icecream import ic

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
    :return: magitude and flux in filter filterout
    """
    passband = ss.Passband(filterin)
    sed.normalise_to_mag(magin, passband)
    passband_ = ss.Passband(filterout)
    flux_ = sed.calc_flux(passband_)
    mag_ = sed.calc_mag(passband_, system='AB')
    flux_nu_=ss.flambda_to_fnu(passband_.effective_wa, flux_)
    return mag_,flux_nu_

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
path_to_angles = ['/Users/maxmen3/stiva/pietro_models/',
                  '/Users/maxmen3/stiva/pietro_models/',
                  '/Users/maxmen3/stiva/pietro_models/',
                  '/Users/maxmen3/stiva/clusters/PSZ1G311/delens/',
                  '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/maxmen3/stiva/pietro_models/',
                  '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/maxmen3/stiva/gabriel_models/best_fit/',
                  '/Users/maxmen3/stiva/gabriel_models/best_fit/',
                  '/Users/maxmen3/stiva/gabriel_models/best_fit/',
                  '/Users/maxmen3/stiva/gabriel_models/best_fit/']


# redshift delle lenti
zl_ = [0.439,0.396,0.348,0.4436,0.375,0.308,0.545,0.543,0.45,0.352,0.587,0.234]

icl = 5 #2
zs = 3.98 #1.0
print ('creating deflector...')
df = lst.create_deflector(parfile = path_to_angles[icl] + cluster[icl] + '.par',
                          filex = path_to_angles[icl] + cluster[icl] + '_angx.fits',
                          filey = path_to_angles[icl] + cluster[icl] + '_angy.fits',
                          zl= zl_[icl], zs = zs, zsnorm = 1.0, resc_fact = 1.0)

print ('cosmological parameters: H0=%s, Om0=%s, OmL-%s' % (df.co.H0, df.co.Om0, df.co.Ode0))
co = df.co

sx,sy,fovsp = df.fovSP()

# set up source and observe it
magn = 24.0
npix=1000 #df.a1.shape[0]#500
fsize = 100.0 #np.max(df.thetax)-np.min(df.thetax)
pixel = fsize / (npix - 1)
ob = observation(size=fsize, Npix=npix, zp=23.9, texp=2500, bkg=22.0)
flgal = ob.mag2counts(magn)
fsub = 0.2

# Assign a SED to the host galaxy: here we assume the host has a SED typycal of a spiral galaxy
sed = getSed(sedm,zs=zs,imin=5,imax=7)

for filtern_ in filtern_out:
    mag__, flux__ = getMagFlux(sed,filterin=filtern,magin=magn,filterout=filtern)

kwargs = {
    'n': 1.0,
    'q': 0.2,
    'ys1': 28.3951950, #5.5,  # ys1+dy1,
    'ys2': 40.2729467, #2.1,  # ys2+dy2,
    'pa': np.pi/4.0,
    're': 0.3,
    'flux': (1.0-fsub)*flgal,
    'zs': zs
}



sizex = [-fsize/2.0, fsize/2.0]
sizey = [-fsize/2.0, fsize/2.0]
se = sersic(size=fsize, Npix=npix, gl=df, save_unlensed=True, rmaxf=10.0, **kwargs)



l=np.linspace(0.001,1.0,10000)

# dependence on x_cut
plotThisFigure=False

if plotThisFigure:
    fig,ax=plt.subplots(1,4,figsize=(20,5))

    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=0.01)
    ax[0].plot(l,fl/np.max(fl),'-',label='1')
    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=0.5)
    ax[0].plot(l,fl/np.max(fl),'-',label='2')
    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=1.0)
    ax[0].plot(l,fl/np.max(fl),'-',label='3')

    # dependence on delta
    fl = sa.PowerLawExpCut(l,delta=0.01,xcut=1.0)
    ax[1].plot(l,fl/np.max(fl),'-',label='1')
    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=1.0)
    ax[1].plot(l,fl/np.max(fl),'-',label='2')
    fl = sa.PowerLawExpCut(l,delta=10.0,xcut=1.0)
    ax[1].plot(l,fl/np.max(fl),'-',label='3')

    # dependence on gamma
    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=1.0,gamma=0.3)
    ax[2].plot(l,fl/np.max(fl),'-',label='1')
    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=1.0,gamma=3)
    ax[2].plot(l,fl/np.max(fl),'-',label='2')
    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=1.0,gamma=30)
    ax[2].plot(l,fl/np.max(fl),'-',label='3')

    # dependence on beta
    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=1.0,beta=-1)
    ax[3].plot(l,fl/np.max(fl),'-',label='1')
    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=1.0,beta=-2)
    ax[3].plot(l,fl/np.max(fl),'-',label='2')
    fl = sa.PowerLawExpCut(l,delta=1.0,xcut=1.0,beta=-3)
    ax[3].plot(l,fl/np.max(fl),'-',label='3')

    ax[0].set_title('$x_{cut}=0.01,0.5,1.0$')
    ax[1].set_title('$\delta=0.01,1,10$')
    ax[2].set_title('$\gamma=0.3,3,30$')
    ax[3].set_title('$\\beta=-1,-2,-3$')
    ax[0].legend()
    for i in range(4):
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
        ax[i].set_ylim(0.0001,1.1)
        ax[i].set_xlabel('x')
    ax[0].set_ylabel('f(x)')
    plt.show()

#l = sa.samplePowerLawExCut(delta=1.0,xcut=0.2,gamma=5,nx=100000,xmin=0.5,fsub=1.0)
l_ = sa.samplePowerLawExCut(xmin=0.0001,xcut=0.01,fsub=0.2,delta=1.0,beta=-2.0,gamma=3.0)
ic(np.sum(l_))

l_ = np.array(l_)
isel = l_> 0.0001
l = l_[isel]

#ic(l)

image=se.image_unlensed.copy()
image[image<(image.max()*0.05)]=0.0
x,y = sa.sample2Dimage(image.T,n=len(l))
x_arcsec, y_arcsec = (x-npix/2.0+0.5) * pixel, (y-npix/2.0+0.5) * pixel

plotThisFigure=False
if plotThisFigure:
    fig,ax=plt.subplots(1,1,figsize=(20,20))
    ax.imshow(se.image_unlensed,origin='lower',extent=[-fsize/2.,fsize/2.,-fsize/2.,fsize/2.])
    ax.plot(x_arcsec,y_arcsec,'o',color='red',alpha=0.8)
    ax.plot(se.ys1,se.ys2,'o',color='yellow')
    plt.show()

image = se.image_unlensed.copy()
limage = se.image.copy()

lsub = l * flgal

from tqdm import tqdm

for i in tqdm(range(len(l))):
    kwargs_sub = {
        'n': 4.0,
        'q': 1.0,
        'ys1': x_arcsec[i],  # ys1+dy1,
        'ys2': y_arcsec[i],  # ys2+dy2,
        'pa': 0.0,
        're': 0.1*(l[i]/(0.01))**(1./3.),
        'flux': l[i]*flgal,
        'zs': zs
    }
    se_sub = sersic(size=fsize, Npix=npix, gl=df, save_unlensed=True, rmaxf=10.0, **kwargs_sub)
    image= image + se_sub.image_unlensed.copy()
    limage = limage + se_sub.image.copy()

tl=df.tancl() #critica tang
rl=df.radcl() #critica rad
if rl.size > 0:
    rcl=df.getCaustics(rl)
if tl.size > 0:
    tcl=df.getCaustics(tl)

fig,ax=plt.subplots(1,1,figsize=(20,20))
#image[image<(image.max()*0.05)]=0.0
logimage=np.log10(image+1)
immax = np.max(logimage)
ax.imshow(np.log10(image+1),origin='lower',vmax=immax*0.4,cmap='cubehelix',
          extent=[-fsize/2.,fsize/2.,-fsize/2.,fsize/2.])
#ax.plot(x,y,'o',color='red',alpha=0.8)
for t in tcl:
    u,v = df.getCausticPoints(t)
    ax.plot(u,v,'-',color='yellow')

for r in rcl:
    m,n = df.getCausticPoints(r)
    ax.plot(m,n,'-',color='yellow')

ax.set_xlim([-fsize/2.,fsize/2.])
ax.set_ylim([-fsize/2.,fsize/2.])

ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.set_aspect('equal')
#plt.show()

fig.savefig('clumpygal.pdf',bbox_inches='tight',dpi=300)


fig,ax=plt.subplots(1,1,figsize=(20,20))
#image[image<(image.max()*0.05)]=0.0
logimage=np.log10(limage+1)
immax = np.max(logimage)
ax.imshow(logimage,origin='lower',vmax=immax*0.4,cmap='cubehelix',
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
