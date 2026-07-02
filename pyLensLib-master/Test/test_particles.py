from pyLensLib.piemd import piemd
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from astropy import constants as const
import astropy.units as units

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

co = FlatLambdaCDM(Om0=0.3,H0=70.)

kwargs = {'theta_c': 2.0, 'theta_t': 1000.0, 'zl': 0.4, 'zs': 2.0, 'sigma0': 1300.0}
kwargs_sub = {'x1': 30.0, 'x2': 30.0, 'theta_c': 0.001, 'theta_t': 20.0, 'zl': 0.4, 'zs': 2.0, 'sigma0': 150.0}
pm = piemd(co,**kwargs)
pm_sub = piemd(co,**kwargs_sub)
dx=np.deg2rad(pm_sub.x1/3600.0)*pm.dl.value
dy=np.deg2rad(pm_sub.x2/3600.0)*pm.dl.value

def generateRad(pm,rmin=1e-5,rmax=2.0,mpart=1e9):
    r=np.logspace(np.log10(rmin),np.log10(rmax),15000)
    m2d=pm.m2Dr(r)
    npart = int(np.max(m2d)/mpart)
    print (npart)
    m2d_n = m2d / np.max(m2d)
    rn = np.random.uniform(0, 1, npart)
    r_all = []
    for rn_ in rn:
        if rn_ < r[0]:
            r_all.append(r[0])
        else:
            index_min = np.argmin(np.abs(rn_ - m2d_n))
            r_all.append(r[index_min])

    return np.array(r_all)

def r2xy(r):
    pa = np.random.uniform(0.0, 2.0 * np.pi, r.size)
    x = r * np.cos(pa)
    y = r * np.sin(pa)
    return x,y

rmin=1e-5
rmax=1.0
r=np.logspace(np.log10(rmin),np.log10(rmax),15000)
s2d=pm.surf_density(r)
plt.plot(r,s2d)
plt.xlim([1e-2,2.0])
plt.xscale('log')
plt.yscale('log')
plt.show()



# throw npart particles
pmass = 5e8
r_all = generateRad(pm,mpart=pmass)
r_sub = generateRad(pm_sub,mpart=pmass)
x_, y_ = r2xy(r_all)
x_sub, y_sub = r2xy(r_sub)
x_sub = x_sub+dx
y_sub = y_sub+dx



# select particles in a smaller region:
rmax = 0.5
isel = (np.abs(x_) <= rmax) & (np.abs(y_) <= rmax)
isel_sub = (np.abs(x_sub) <= rmax) & (np.abs(y_sub) <= rmax)
x=x_[isel]
y=y_[isel]
xs=x_sub[isel_sub]
ys=y_sub[isel_sub]
xt=np.append(x,xs)
yt=np.append(y,ys)


print ('selecting %10.4e particles' % (xt.size))

# ADDING PARTICLES ON THE CORNERS OF TEH MAP FOR CENTERING ISSUES
xt=np.append(xt,-rmax)
yt=np.append(yt,-rmax)
xt=np.append(xt,rmax)
yt=np.append(yt,rmax)

# total mass and particle mass
mtot=pmass*(xt.size-2)
mass=np.ones(xt.size)*pmass
mass[-1]=0.0
mass[-2]=0.0
print (("%10.4e %10.4e") % (mtot,mass.sum()))

npix=1024

xi = np.linspace(-rmax, rmax, npix+1)
yi = np.linspace(-rmax, rmax, npix+1)
sigma=4.0
massmap, xedges, yedges = np.histogram2d(yt, xt, bins=(xi, yi), weights=mass)
img = ndimage.gaussian_filter(massmap, sigma=(sigma, sigma), order=0)
#img = img/pix**2/sigma_crit(pm)
plt.imshow(np.log10(img+1e-2),origin='lower')
plt.show()

xmax=rmax
pix=xmax*2.0/(npix-1)

# assign a third coordinate
zt = np.ones(len(xt))
pos = np.stack((xt,yt,zt),axis=1)

"""
# create a mass map: using nb neighbors and max_hsml=0.1 (100 kpc)
nb=500
max_hsml=0.5

from sphviewer.tools import QuickView
qv = QuickView(pos, mass= mass, r='infinity', nb=nb, plot=False,
               xsize=npix, ysize=npix, logscale=False,
               max_hsml=max_hsml)

img=qv.get_image()
print (img.sum(),mtot)
img=img/np.sum(img)*np.sum(mass)
"""

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img, extent=[-rmax, rmax, -rmax, rmax],origin='lower')
plt.show()

# perform ray-tracing: shoot Nray rays on a fov=200
from pyLensLib.raytracer import raytracer
# the mass map fov in arcsec
fov=np.rad2deg(2.0*rmax/pm.dl.value)*3600.0
kwargs_rt = {'zl': 0.4, 'zs': 2.0, 'fov':fov}
# shooting Nray x Nray rays on FOVray x FOVray fov
FOVray=100
rt=raytracer(co,img,Nray=2048,FOVray=FOVray,fromfile=False,**kwargs_rt)

# set up the deflector:
from pyLensLib.deflector import deflector
df = deflector(co, angx=rt.a1, angy=rt.a2, **kwargs_rt)
theta = np.linspace(-FOVray/2., FOVray/2., rt.a1.shape[0])
df.setGrid(theta)
pm.setGrid(theta)
pm_sub.setGrid(theta)
pm.combinewith(pm_sub)
tl = df.tancl()
rl = df.radcl()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(df.ka, extent=[-FOVray/2., FOVray/2., -FOVray/2., FOVray/2.],vmax=3.0,origin='lower')

for t in tl:
    xt, yt = df.getCritPoints(t)
    ax.plot(xt, yt, '-', color='white')

for t in rl:
    xt, yt = df.getCritPoints(t)
    ax.plot(xt, yt, '-', color='white')

tl = pm.tancl()
rl = pm.radcl()
for t in tl:
    xt, yt = pm.getCritPoints(t)
    ax.plot(xt, yt, '--', color='yellow')

for t in rl:
    xt, yt = pm.getCritPoints(t)
    ax.plot(xt, yt, '--', color='yellow')

plt.show()
