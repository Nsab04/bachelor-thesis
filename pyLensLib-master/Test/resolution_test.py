import numpy as np
import matplotlib.pyplot as plt
import pyLensLib.lenstool as lst
from sphviewer.tools import QuickView

cluster = ['S1063','M0416','M1206pl','PSZ1G311_200',
           'A370','A2744','M0717','M1149',
           'M0329','M1931','M2129','R2129']

# posizione delle mappe degli angoli di deflessione
path_to_angles = ['/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/stiva/clusters/PSZ1G311/delens/',
                  '/Users/massimo/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/massimo/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/']


# redshift delle lenti
#zl_ = [0.439,0.396,0.348,0.4436,0.375,0.308,0.545,0.543,0.45,0.352,0.587,0.234]

icl = 3
zs = 2.5

zl_s = lst.getLensRedshift(path_to_angles[icl] + cluster[icl] + '.par')
print (path_to_angles[icl] + cluster[icl] + '.par')
print (('Working with cluster %s at redshift %s') % (cluster[icl], zl_s))
zl = float(zl_s)

print ('creating deflector...')
df = lst.create_deflector(parfile = path_to_angles[icl] + cluster[icl] + '.par',
                          filex = path_to_angles[icl] + cluster[icl] + '_angx.fits',
                          filey = path_to_angles[icl] + cluster[icl] + '_angy.fits',
                          zl= zl, zs = zs, zsnorm = 1.0, resc_fact = 1.0, compute_potential=True)

co = lst.getCosmo(best_par= path_to_angles[icl] + cluster[icl] + '.par')

from pyLensLib import samplers as sp


def sigma_cr(df):
    """
    Calculates the critical surface density for lensing
    :return:  critica surface density for the lens, given its redshift and the source redshift
    """
    from astropy.constants import c, G
    from astropy import units as u
    c2_G_Msun_Mpc = (c ** 2 / G).to(u.Msun / u.Mpc)
    sigma_cr = c2_G_Msun_Mpc / (4 * np.pi) * (df.ds / df.dl / df.dls)
    print (sigma_cr)
    return (sigma_cr.value)

pixel = np.deg2rad(df.pixel_scale/3600.0)*df.dl.value
print (('Total mass=%10.4e') % (np.sum(df.ka*sigma_cr(df))*pixel**2))

mass = np.sum(df.ka*sigma_cr(df))*pixel**2
npart = 3000000
mp = mass/npart
print (('mp=%10.4e') % (mp))


x, y = sp.sample2Dimage(df.ka,npart)

# convert to Mpc
x = x * pixel - (df.nray1-1) * pixel / 2.0
y = y * pixel - (df.nray2-1) * pixel / 2.0

from scipy import stats
print(stats.describe(x))
#exit()

fov = (df.nray1 - 1) * pixel

#plt.imshow(df.ka,origin='lower',vmax=2.0,extent=[-fov/2.,fov/2.,-fov/2.,fov/2.])
#plt.plot(x,y,',',color='red')
#plt.show()

#bins = np.linspace(0,df.ka.shape[0]-1,df.ka.shape[0])
#map, xe, ye = np.histogram2d(x,y,bins=bins)
#map*=mp
#plt.imshow(map,origin='lower')
#plt.show()

fig,ax = plt.subplots(1,3,figsize=(25,10))
ax[0].imshow(df.ka, origin='lower', vmax=2.0, extent=[-100,100,-100,100])
# display lenstool model and critical lines
tl=df.tancl()
rl=df.radcl()
ctl=df.getCaustics(tl)
crl=df.getCaustics(rl)
for c in tl:
    xc,yc=df.getCritPoints(c,pixel_units=False)
    if (c.principale):
        ax[0].plot(xc,yc,'-',color='white')
    else:
        ax[0].plot(xc,yc,'-',color='yellow')
for c in rl:
    xc,yc=df.getCritPoints(c,pixel_units=False)
    ax[0].plot(xc,yc,'-',color='white')

### build deflector from sampled particle distribution
m=np.ones(npart)*mp
print ('test',len(m),mass, np.sum(m))
pos = np.zeros((len(x),3))
pos[:,0] = y
pos[:,1] = x
pos[:,2] = np.random.normal(0.0,0.01,len(x))
min_hsml=0.001
max_hsml=0.1
nb=100

print ('minmax x',np.min(x),np.max(x))
print(stats.describe(x))

qv = QuickView(pos, m, r='infinity', nb=nb, plot=False, xsize=2048, ysize=2048, logscale=False,
                       min_hsml=min_hsml, max_hsml=max_hsml)
        #qv=QuickView(pos.T, mass, r='infinity', nb=nb, plot=False, xsize=npix, ysize=npix, logscale=False)
img=qv.get_image()
img=img/img.sum()*m.sum()

from pyLensLib.raytracer import raytracer
from  pyLensLib.deflector import deflector

fov_ray_arcsec = 200
nray=2048
kwargs = {'zl': zl, 'zs': 3.0, 'fov': fov_ray_arcsec}
rayt=raytracer(co,img,Nray=nray,FOVray=fov_ray_arcsec,fromfile=False,**kwargs)
kwargs_def={'zl': zl, 'zs': kwargs['zs']}
df_simu=deflector(co,angx=rayt.a1,angy=rayt.a2,**kwargs_def)
theta=np.linspace(-fov_ray_arcsec/2.0,fov_ray_arcsec/2.0,nray)
df_simu.setGrid(theta=theta,compute_potential=False)
df_simu.ka[df_simu.ka<0]=0.0

print (m.sum(),img.max(),img.min())
ax[1].imshow(img,origin='lower',vmax=np.max(img)*0.5,extent=[-fov_ray_arcsec/2.,fov_ray_arcsec/2.,-fov_ray_arcsec/2.,fov_ray_arcsec/2.])

tl=df_simu.tancl()
rl=df_simu.radcl()
ctl=df_simu.getCaustics(tl)
crl=df_simu.getCaustics(rl)
for c in tl:
    xc,yc=df_simu.getCritPoints(c,pixel_units=False)
    if (c.principale):
        ax[1].plot(xc,yc,'-',color='white')
    else:
        ax[1].plot(xc,yc,'-',color='yellow')
for c in rl:
    xc,yc=df_simu.getCritPoints(c,pixel_units=False)
    ax[1].plot(xc,yc,'-',color='white')

ax[2].imshow(img,origin='lower',vmax=np.max(img)*0.5,extent=[-fov_ray_arcsec/2.,fov_ray_arcsec/2.,-fov_ray_arcsec/2.,fov_ray_arcsec/2.])
plt.show()