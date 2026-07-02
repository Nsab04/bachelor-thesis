from pyLensLib.cluster import cluster
from pyLensLib.subfind import subfind
from pyLensLib.raytracer import raytracer
import numpy as np
import astropy.io.fits as fits
from pyLensLib.deflector import deflector
cluster_file = "snap_058"
import matplotlib.pyplot as plt


# STEP 1: MASS MAP CREATION
# Setting up the path to the gadget3 snapshot file
snapshot = '/Users/massimo/projects/ggsl/testing_codes/massmaps_python/BH_2015/snap_058'
# also setting the path to a subfind file. This file contains the halo center coordinates.
# these coordinates can also be provided manually (see later)
subfind_file = "/Users/massimo/stiva/Dianoga/new_elena/Subfind_AGN/D1/sub_058.0"

# read the center positions from the subfind file
sf=subfind(subfind_file)
xc,yc,zc=sf.gpos[0]

#
# create a mass map reading the particle positions from the gadget3 file.

# Before reading the particle positions and masses, get some infos from the header of the gadget file
# Specifically, the cluster class constructor reads the cosmological parameters and builds
# LambdaCDM model (astropy), which becomes an attribute of the cluster instance
cl = cluster(snapshot=snapshot,parttype=[0],xc=xc,yc=yc,zc=zc)

npix=2048 # number of pixels in the mass map (on each side)
img = np.zeros((npix,npix))
fov=400.0 # mass map side length in arcsec
fov_mpc=np.deg2rad(fov/3600.0) * cl.co.angular_diameter_distance(cl.zl).value # now converted to Mpc

parttype = [1]#[0,1,4,5] # particle types to be used to build the mass map
for itype in parttype:
    cl=cluster(snapshot=snapshot,parttype=[itype],xc=xc,yc=yc,zc=zc,IT=np.array([[1,0,0],[0,0,1],[0,1,0]]),alignIT=True)
    this_img = cl.massMapSPH(nb=50,pX=90.,pY=90,npix=npix,xmin=-fov_mpc/2.0,
                             xmax=fov_mpc/2.0,ymin=-fov_mpc/2.0,ymax=fov_mpc/2.0,
                             zmin=-10.0,zmax=10.0)
    img = img.copy() + this_img

print ('mass %10.4e' % img.sum())

# display the mass map
#import matplotlib.pyplot as plt
#fig,ax=plt.subplots(1,1,figsize=(10,10))
#ax.imshow(np.sqrt(img),origin='lower')
#plt.show()

# I am now assuming that you want to save the mass maps in some fits file:
primary_hdu = fits.PrimaryHDU(img)
primary_hdu.header['ZL'] = cl.zl
primary_hdu.header['CDELT1'] = -fov/img.shape[0]/3600.0
primary_hdu.header['CDELT2'] = fov/img.shape[0]/3600.0
hdul = fits.HDUList([primary_hdu])
hdul.writeto('mass_map.fits',overwrite=True)

# STEP 2: RAY-TRACING
# Let assume that you want to perform the lensing analysis of the mass map that you created
# first, create a dictionary containing the lens and source redshifts and the map size in arcsec
kwargs = {'zl': cl.zl, 'zs': 3.0, 'fov': fov}
# then, create the deflection angle maps by raytracing through the mass map. The raytracer class allows to
# shoot rays through an arbitrary region on the input mass map. You can define this region by
# setting the FOVray parameter. In this example, I choose to ray-trace through the central
# 200x200 arcsec region of the mass map. The ray grid has Nray x Nray pixels. Note that the center
# of the ray grid coincides with the center of the mass map.
FOVray=200.0
nray=2048
rayt=raytracer(cl.co,'mass_map.fits',Nray=nray,FOVray=FOVray,fromfile=True,**kwargs)

# it is also possible to pass to the raytracer constructor the mass map as a numpy array instead of a
# fits file:
#rayt=raytracer(cl.co,img,Nray=2048,FOVray=200.0,fromfile=False,**kwargs)

# Now we can build a deflector. We need to provide another dictionary containing the lens and source redhifts
kwargs_def={'zl': cl.zl, 'zs': kwargs['zs']}
# The deflector can be built either using the rayt potential or the deflection angle maps:
#df=deflector(cl.co,pot=rayt.pot,usePotential=True,**kwargs_def)
df=deflector(cl.co,angx=rayt.a1,angy=rayt.a2,**kwargs_def)

# Once the deflector is created, we can compute several lensing maps (convergence, shear, etc).
# we can do that by making this call
theta = np.linspace(-FOVray/2.0,FOVray/2.0,nray)
df.setGrid(theta=theta)

# the maps can be accessed as attributes of df:
"""
fig,ax=plt.subplots(3,2,figsize=(30,20))
ax[0,0].imshow(df.a1,origin='lower',extent=[-FOVray/2.0,FOVray/2.0,-FOVray/2.0,FOVray/2.0])
ax[0,1].imshow(df.a2,origin='lower',extent=[-FOVray/2.0,FOVray/2.0,-FOVray/2.0,FOVray/2.0])
ax[1,0].imshow(df.ka,origin='lower',extent=[-FOVray/2.0,FOVray/2.0,-FOVray/2.0,FOVray/2.0])
mu = 1.0/((1.0-df.ka)**2-df.g1**2-df.g2**2)
ax[1,1].imshow(np.log10(np.abs(mu)),origin='lower',extent=[-FOVray/2.0,FOVray/2.0,-FOVray/2.0,FOVray/2.0])
ax[2,0].imshow(df.g1,origin='lower',extent=[-FOVray/2.0,FOVray/2.0,-FOVray/2.0,FOVray/2.0])
ax[2,1].imshow(df.g2,origin='lower',extent=[-FOVray/2.0,FOVray/2.0,-FOVray/2.0,FOVray/2.0])
ax[0,0].set_title(r'$\alpha_1$')
ax[0,1].set_title(r'$\alpha_2$')
ax[1,0].set_title(r'$\kappa$')
ax[1,1].set_title(r'$\mu$')
ax[2,0].set_title(r'$\gamma_1$')
ax[2,1].set_title(r'$\gamma_2$')
plt.show()
"""

# STEP 3: CRITICAL LINES AND CAUSTICS
# To find the critical lines, you can use the tancl and radcl methods:
# tangential critical lines:
tl = df.tancl()
# radial critical lines:
rl = df.radcl()

# these functions return lists of CriticalLine and Caustic objects (see critcau.py), ordered by size.
# For example, in the case of a lens containing substructures, the first critical line in the list is
# probably that of the main halo, while the other critical lines are those generated by the subhalos.


# the critical points can be accessed using the function getCritPoints applied to each critical line:
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(df.ka,extent=[-FOVray/2.0,FOVray/2.0,-FOVray/2.0,FOVray/2.0],origin='lower')
for t in tl:
    x, y = df.getCritPoints(t)
    ax.plot(x,y,'-',color='white')

for r in rl:
    x, y = df.getCritPoints(r)
    ax.plot(x, y, '-', color='orange')

# if you want to find the caustics, the procedure is as follows:
# for each critical line, use the function getCaustics:
ctl=df.getCaustics(tl)
crl=df.getCaustics(rl)

# then use the function getCausticPoints to access the individual caustic points:
for t in ctl:
    x, y = df.getCausticPoints(t)
    ax.plot(x,y,'--',color='red')

for r in crl:
    x, y = df.getCausticPoints(r)
    ax.plot(x, y, '--', color='blue')

plt.show()

# to get the critical line size, you can use the function thetaEv. This function returns an array whose elements are
# the equivalent Einstein radii of all critical lines.

thetaE = df.thetaEv(clt=tl)

# For example, the Einstein radius of the main critical line is:
print ('Main C.L. size (Eqv. Einstein radius):',thetaE[0])