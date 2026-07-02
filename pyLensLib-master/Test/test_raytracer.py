from pyLensLib.cluster import cluster
from pyLensLib.raytracer import raytracer
from pyLensLib.subfind import subfind
from pyLensLib.deflector import deflector
from pyLensLib.observation import observation
from pyLensLib.sersic import sersic
import numpy as np
import astropy.io.fits as fits

import matplotlib

matplotlib.use('TkAgg')

snapshot = 'snap_058'
subfind_file = "sub_058.0"
sf=subfind(subfind_file)
xc,yc,zc=sf.gpos[0]
cl = cluster(snapshot=snapshot,parttype=[1],xc=xc,yc=yc,zc=zc)

import matplotlib.pyplot as plt
#kwargs = {'zs': 3.0}
#fov_mpc = 1.22183607506*2.0
#fov = fov_mpc / cl.co.angular_diameter_distance(cl.zl).value / np.pi * 180. * 3600.

test_from_array = True
if test_from_array:
    hdul=fits.open('mass_map.fits')
    #hdul=fits.open('/Users/massimo/stiva/Dianoga/new_elena/AGN_400_2048_2/lensing/map_058_1_2_sph.fits')
    massmap=hdul[0].data
    fov = hdul[0].header['NAXIS1']*hdul[0].header['CDELT2']*3600.0
    fov_mpc = fov *  cl.co.angular_diameter_distance(cl.zl).value * np.pi / 180. / 3600.
    fov_ray = fov * 0.9
    print (fov,fov_mpc)
    kwargs = {'zl': cl.zl, 'zs': 3.0, 'fov': fov}
    rayt=raytracer(cl.co,massmap,Nray=2048,FOVray=fov_ray,fromfile=False,**kwargs)
else:
    kwargs = {'zl': cl.zl, 'zs': 3.0, 'fov': fov}
    rayt=raytracer(cl.co,'mass_map.fits',Nray=2048,FOVray=fov/2.0,fromfile=True,**kwargs)

kwargs_def={'zl': cl.zl, 'zs': kwargs['zs']}
print (kwargs_def)
df=deflector(cl.co,angx=rayt.a1,angy=rayt.a2,**kwargs_def)
#df=deflector(cl.co,pot=rayt.pot,usePotential=True,**kwargs_def)
theta=np.linspace(-fov_ray/2.,fov_ray/2.,2048)
print (theta.min(),theta.max(),theta[1]-theta[0])
df.setGrid(theta=theta,compute_potential=True)

#save deflection angle maps in two separate fits files
# first file for the x component
hdul = fits.HDUList([fits.PrimaryHDU(df.a1),])
hdul.writeto('deflx.fits',overwrite=True)
# second file for the y component
hdul = fits.HDUList([fits.PrimaryHDU(df.a2)])
hdul.writeto('defly.fits',overwrite=True)

"""
fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(rayt.mass,origin='low',extent=[-fov/2,fov/2.,-fov/2.,fov/2])
cl=df.tancl()
for c in cl:
    x,y=df.getCritPoints(c,pixel_units=False)
    ax.plot(x,y,'-',color='white')
    #print (np.min(x),np.max(x))

ax.set_xlim([-fov_ray/2.0,fov_ray/2.0])
ax.set_ylim([-fov_ray/2.0,fov_ray/2.0])
plt.show()
"""
beta=[15.974,-8.57143]
beta=[0.0,0.0]
from pyLensLib.pointsrc import pointsrc


fig,ax=plt.subplots(1,1,figsize=(10,10))
#ax.imshow(tds,origin='low',extent=[-100,100,-100,100])

mag_source=23.0
ob = observation(size=200.0, Npix=1000, zp=23.9, texp=565*4, bkg=22.0)
fl_gal=ob.mag2counts(mag_source)
kwargs_source_light={
    'n': 1.0, # sersic index
    're': 0.5, # effective radius
    'q': 1.0, # axis ratio
    'pa': np.pi/4., # position angle
    'ys1': beta[0], # position x (on the source plane)
    'ys2': beta[1], # position y (on the lens plane)
    'flux': fl_gal, # flux of the source (in counts/s)
    'zs': 5.8#kwargs['zs']
}



#print (df.multImaCrossSection())
#print (df.ggslCrossSection(minsize=0.9,maxsize=3.0,dmax=100.0))
print ('Redshift of the source (original):',df.zs)
df.change_redshift(kwargs_source_light['zs'])
print ('Redshift of the source (new):',df.zs)

kwargs_psr = {
    'zs': df.zs,
    'ys1': beta[0],
    'ys2': beta[1]
}

ps=pointsrc(size=200.0, sizex=None, sizey=None, Npix=1000, gl=df, **kwargs_psr)
xi,yi,mui=ps.find_images()

se=sersic(size=200.0, Npix=1000,gl=df,**kwargs_source_light)
#df.change_redshift(kwargs_source_light['zs'])
#print (df.multImaCrossSection())
#print (df.ggslCrossSection(minsize=0.9,maxsize=3.0,dmax=100.0))

tds=df.t_delay_surf(beta=beta)
ax.imshow(se.image,origin='lower',extent=[-100,100,-100,100],vmax=se.image.max()*0.5)
ax.contour(np.sqrt((tds-tds.min())/(tds.max()-tds.min())),levels=np.logspace(-2,0,30),
           extent=[theta.min(),theta.max(),theta.min(),theta.max()],colors='white',alpha=0.2)
print (theta.min(),theta.max(),theta.min(),theta.max())
tl=df.tancl()
rl=df.radcl()
ctl=df.getCaustics(tl)
crl=df.getCaustics(rl)
for c in tl:
    x,y=df.getCritPoints(c,pixel_units=False)
    if (c.principale):
        ax.plot(x,y,'-',color='white')
    else:
        ax.plot(x,y,'-',color='yellow')
for c in rl:
    x,y=df.getCritPoints(c,pixel_units=False)
    ax.plot(x,y,'-',color='white')

totarea=0.0
for c in ctl:
    x,y=df.getCausticPoints(c)
    if (c.principale):
        ax.plot(x,y,'-',color='orange',alpha=0.3)
    else:
        ax.plot(x, y, '-', color='red', alpha=0.3)
    totarea+=c.area*df.pixel_scale**2

print ('totarea', totarea, df.pixel_scale, df.nray1*df.pixel_scale, len(ctl))

for c in crl:
    x,y=df.getCausticPoints(c)
    ax.plot(x,y,'-',color='orange',alpha=0.3)
#ax.imshow(tds-tds.min(),origin='lower',extent=[-fov_ray/2,fov_ray/2,-fov_ray/2,fov_ray/2.0],alpha=0.001)
ax.plot(beta[0],beta[1],'o',color='red')
ax.plot(xi,yi,'+',color='red')
ax.set_ylim([-101,101])
ax.set_xlim([-101,101])
plt.show()

thetae=df.thetaEv()
fig,ax = plt.subplots(1,1,figsize=(10,10))
theta_bins=np.linspace(0.0,30.0,60)
ax.hist(thetae,bins=theta_bins)
ax.set_xlabel(r'$\theta_E$')
ax.set_ylabel(r'$N(\theta_E)$')
plt.show()

