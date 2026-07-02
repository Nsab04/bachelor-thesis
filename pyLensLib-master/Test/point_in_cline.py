"""
This script allows to check if a point is contained within a critical line object
"""

from pyLensLib.piemd import piemd
from astropy.cosmology import FlatLambdaCDM
import numpy as np
from shapely.geometry import  MultiPoint, Point

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# set up cosmology
co = FlatLambdaCDM(H0=70.0, Om0=0.3)

# lens and source redshifts
zl = 0.5
zs = 1.0

# properties of the lens
kwargs = {'zl': zl,  # redshift
          'zs': zs,  # source redshift (all lensing properties are computed for a source at this redshift
          'sigma0': 1300.0,  # lens central velocity dispersion
          'q': 0.6,  # lens axis ratio b/a
          'pa': -np.pi / 4.0,  # lens position angle
          'theta_c': 10.0,  # lens core radius
          'theta_t': 1000.0,  # lens truncation radius
          'x1': 0.0,  # lens position x
          'x2': 0.0}  # lens position y

# properties of the lens
kwargs2 = {'zl': zl,  # redshift
          'zs': zs,  # source redshift (all lensing properties are computed for a source at this redshift
          'sigma0': 300.0,  # lens central velocity dispersion
          'q': 0.6,  # lens axis ratio b/a
          'pa': -np.pi / 4.0,  # lens position angle
          'theta_c': 0.1,  # lens core radius
          'theta_t': 100.0,  # lens truncation radius
          'x1': 25.0,  # lens position x
          'x2': 0.0}  # lens position y

# properties of the lens
kwargs3 = {'zl': zl,  # redshift
          'zs': zs,  # source redshift (all lensing properties are computed for a source at this redshift
          'sigma0': 300.0,  # lens central velocity dispersion
          'q': 0.6,  # lens axis ratio b/a
          'pa': -np.pi / 4.0,  # lens position angle
          'theta_c': 0.1,  # lens core radius
          'theta_t': 100.0,  # lens truncation radius
          'x1': 22.0,  # lens position x
          'x2': 22.0}  # lens position y

# create an instance of the lens
dpie = piemd(co, **kwargs)
dpie2 = piemd(co, **kwargs2)
dpie3 = piemd(co, **kwargs3)

npix_grid = 1000
sz = 100.0
theta = np.linspace(-sz / 2., sz / 2., npix_grid)
dpie.setGrid(theta)
dpie2.setGrid(theta)
dpie3.setGrid(theta)
dpie.combinewith(dpie2)
dpie.combinewith(dpie3)

cl = dpie.tancl()

# generate random points:
npoints = 1500
xp = sz*np.random.random_sample(npoints) - sz/2.
yp = sz*np.random.random_sample(npoints) - sz/2.

def conv2pixel(xp,yp,sz,npix_grid):
    xp_ = (xp + sz / 2.) / sz * npix_grid
    yp_ = (yp + sz / 2.) / sz * npix_grid
    return xp_, yp_

def conv2as(xp,yp,sz,npix_grid):
    xp_ = xp * sz / npix_grid - sz / 2.
    yp_ = yp * sz / npix_grid - sz / 2.
    return xp_, yp_

xp, yp = conv2pixel(xp, yp, sz, npix_grid)

points = [(xp[i], yp[i]) for i in range(len(xp))]
pp = [Point(xp[i], yp[i]) for i in range(len(xp))]
p = MultiPoint(points)

import shapely.ops as so

fig,ax =plt.subplots(1,1,figsize=(10,10))
ax.imshow(dpie.ka,origin='lower',extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
xp_, yp_ = conv2as(xp,yp,sz,npix_grid)
ax.plot(xp_,yp_,'o',color="orange")

for c in cl:
    new_shape = so.cascaded_union(c.geometria)
    points_within_c = p.intersection(new_shape)
    isinc = [p.within(new_shape) for p in pp]

    x,y = dpie.getCritPoints(c)
    ax.plot(x,y,color='red')
    if (points_within_c.geom_type == 'Point'):
        if len(points_within_c.coords) >0:
            xpix, ypix = zip(*points_within_c.coords)
            xpix = np.array(xpix)
            ypix = np.array(ypix)
    else:
        xpix = np.array([points_within_c.geoms[i].xy[0][0] for i in range(len(points_within_c.geoms))])
        ypix = np.array([points_within_c.geoms[i].xy[1][0] for i in range(len(points_within_c.geoms))])
    xp_, yp_ = conv2as(xpix,ypix,sz,npix_grid)
    ax.plot(xp_, yp_ ,'o',color="green")
    xx,yy=conv2as(xp,yp,sz,npix_grid)
    ax.plot(xx[isinc],yy[isinc],'+',color='yellow')

#print (list(points_within_c.coords))
plt.show()
#print (p)
#print (new_shape.exterior.coords.xy)
