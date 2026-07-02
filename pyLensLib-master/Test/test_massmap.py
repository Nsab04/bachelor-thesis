from pyLensLib.cluster import cluster
from pyLensLib.subfind import subfind
from pyLensLib.raytracer import raytracer
from pyLensLib.deflector import deflector
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import pyLensLib.gadget  as G
cluster_file = "snap_058"

snapshot = 'snap_058'
subfind_file = "sub_058.0"

mvir=G.read_block(subfind_file,"MVIR")
print ("%10.6e" % (mvir[0]*1e10))


sf=subfind(subfind_file)
xc,yc,zc=sf.gpos[0]
cl = cluster(snapshot=snapshot,parttype=[0],xc=xc,yc=yc,zc=zc)
parttype = [1]#[0,1,4,5]
npix=2048
img = np.zeros((npix,npix))
fov=400.0
fov_mpc=fov * cl.co.angular_diameter_distance(cl.zl).value*np.pi/180./3600.


for itype in parttype:
    cl=cluster(snapshot=snapshot,parttype=[itype],xc=xc,yc=yc,zc=zc,IT=np.array([[1,0,0],[0,0,1],[0,1,0]]),alignIT=True)
    this_img = cl.massMapSPH(nb=20,pX=0.,pY=0.,npix=npix,xmin=-fov_mpc/2.0,xmax=fov_mpc/2.0,ymin=-fov_mpc/2.0,ymax=fov_mpc/2.0,zmin=-10.0,zmax=10.0)

    img = img.copy() + this_img
    print ('mass %10.4e' % this_img.sum())

    #if itype == 4:
    #    primary_hdu = fits.PrimaryHDU(this_img)
    #    hdul = fits.HDUList([primary_hdu])
    #    hdul.writeto('star_map.fits',overwrite=True)


kwargs = {'zl': cl.zl, 'zs': 3.0, 'fov': fov}
nray=2048
fov_ray_arcsec = 200
rayt=raytracer(cl.co,img,Nray=nray,FOVray=fov_ray_arcsec,fromfile=False,**kwargs)
kwargs_def={'zl': cl.zl, 'zs': kwargs['zs']}
df=deflector(cl.co,angx=rayt.a1,angy=rayt.a2,**kwargs_def)
theta=np.linspace(-fov_ray_arcsec/2.0,fov_ray_arcsec/2.0,nray)
df.setGrid(theta=theta,compute_potential=False)
df.ka[df.ka<0]=0.0

print ('mass %10.4e' % img.sum())

import proplot as pplt
import numpy as np
state = np.random.RandomState(51423)
data = state.rand(30, 30).cumsum(axis=1)

# Diverging colormap example
cmap1 = pplt.Colormap('Blues4_r', 'Reds3', name='Diverging', save=True)
cmap2 = pplt.Colormap(
    'Greens1_r', 'Oranges1', 'Blues1_r', 'Blues6',
    ratios=(1, 3, 5, 10), name='SciVisColorUneven', save=True
)

cmap4 = pplt.Colormap(
    'Greys_r', 'Oranges1', 'Blues1_r', 'Blues6',
    ratios=(1, 3, 5, 10), name='SciVisColorUnevenGrey', save=True
)

cmap3 = pplt.Colormap(
    'Greens1_r', 'Oranges1', 'Blues1_r', 'Blues6',
    name='SciVisColorEven', save=True
)
print ('cmap created')


fig,ax=plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(img,origin='lower',cmap=cmap2)
ax[1].imshow(df.ka,origin='lower',vmax=2.0,extent=[-fov_ray_arcsec/2.0,fov_ray_arcsec/2.0,-fov_ray_arcsec/2.0,fov_ray_arcsec/2.0],cmap=cmap2)
tl=df.tancl()
rl=df.radcl()
ctl=df.getCaustics(tl)
crl=df.getCaustics(rl)
for c in tl:
    xc,yc=df.getCritPoints(c,pixel_units=False)
    if (c.principale):
        ax[1].plot(xc,yc,'-',color='white')
    else:
        ax[1].plot(xc,yc,'-',color='yellow')
for c in rl:
    xc,yc=df.getCritPoints(c,pixel_units=False)
    ax[1].plot(xc,yc,'-',color='white')
plt.show()
exit()
#fov=fov_mpc/cl.co.angular_diameter_distance(cl.zl).value/np.pi*180.*3600.
primary_hdu = fits.PrimaryHDU(img)
primary_hdu.header['ZL'] = cl.zl
primary_hdu.header['CDELT1'] = -fov/img.shape[0]/3600.0
primary_hdu.header['CDELT2'] = fov/img.shape[0]/3600.0
hdul = fits.HDUList([primary_hdu])
hdul.writeto('mass_map.fits',overwrite=True)
kwargs = {'zl': cl.zl, 'zs': 3.0, 'fov': fov}
rayt=raytracer(cl.co,'mass_map.fits',Nray=2048,FOVray=200.0,fromfile=True,**kwargs)
fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(rayt.pot,origin='lower')
plt.show()