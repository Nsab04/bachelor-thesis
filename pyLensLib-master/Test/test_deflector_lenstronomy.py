# this script compares the speed of the deflector class in pyLensLib with the
# equivalent functions in lenstronomy
# the lenstronomy implementation is taken from: https://github.com/lenstronomy/lenstronomy-tutorials/blob/main/Notebooks/Clusters/clusters.ipynb


from pyLensLib.deflector import deflector
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import pyLensLib.lenstool as lst
from pyLensLib.pointsrc import pointsrc

zcl = lst.getLensRedshift('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo.par')
print('zcl',zcl)
fov = lst.getFoV('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo.par')
print ('fov',fov,fov[1]-fov[0],fov[3]-fov[2])
fov_ = fov[1]-fov[0]

df = lst.create_deflector('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo.par',
                          filex='/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo_angx.fits',
                          filey='/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo_angy.fits',zl=zcl,zsnorm=1.0,zs=3.0,
                          compute_potential=True)


print (df.co)
# lenstronomy

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import util


from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

ds = cosmo.angular_diameter_distance(1.0)
ds_ = cosmo.angular_diameter_distance(3.0)
dls = cosmo.angular_diameter_distance_z1z2(zcl,1.0)
dls_ = cosmo.angular_diameter_distance_z1z2(zcl,3.0)

print (df.zl)

# make a grid
numPix = df.thetax.shape[0]
deltaPix = (fov[1] - fov[0])/(numPix - 1)

x_grid_interp, y_grid_interp = util.make_grid(numPix, deltaPix)

lens_model_interp = LensModel(lens_model_list=['INTERPOL'])
x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)

# read ax and ay from the fits files
import astropy.io.fits as fits
hdul = fits.open('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo_angx.fits')
a1 = hdul[0].data*ds.value/dls.value*dls_.value/ds_.value
hdul = fits.open('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo_angy.fits')
a2 = hdul[0].data*ds.value/dls.value*dls_.value/ds_.value
hdul = fits.open('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo_pot.fits')
pot = hdul[0].data*ds.value/dls.value*dls_.value/ds_.value

# create lenstronomy ua
kwargs_interp = [{'grid_interp_x': x_axes, 'grid_interp_y': y_axes,
                  'f_': pot,
                  'f_x': a1,
                  'f_y': a2}
                ]

from lenstronomy.Plots import lens_plot
import matplotlib.pyplot as plt
"""
f, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=False, sharey=False)
lens_plot.lens_model_plot(ax, lensModel=lens_model_interp, kwargs_lens=kwargs_interp,
                          numPix=numPix, deltaPix=deltaPix,
                          point_source=False, with_caustics=True, fast_caustic=True, coord_inverse=False)
cl = df.tancl()
for t in cl:
    x,y = df.getCritPoints(t)
    ax.plot(x,y,'--',color='white')



plt.show()
"""


from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
solver = LensEquationSolver(lens_model_interp)
beta_ra_ = -7.40#-0.05
beta_dec_ = -25.29#-14.76

import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData

kwargs_data = sim_util.data_configure_simple(
    numPix,
    deltaPix,
    center_ra=0.0,
    center_dec=0.0,
    inverse=False,
)
pixel_grid = ImageData(**kwargs_data)
#_frame_size = numPix * deltaPix

#x_center, y_center = pixel_grid.center
#delta_pix = pixel_grid.pixel_width
#ra0, dec0 = pixel_grid.radec_at_xy_0
#tranform = pixel_grid.transform_angle2pix
#if (
#        np.linalg.det(tranform) < 0
#):  # if coordiate transform has negative parity (#TODO temporary fix)
#    delta_pix_x = -delta_pix
#else:
#    delta_pix_x = delta_pix
#origin = [ra0, dec0]
kwargs_solver={}

dx = np.linspace(-0.5, 0.5, 5)
dy = np.linspace(-0.5, 0.5, 5)
for i in range(5):
    print ('Iteration:',i,dx[i],dy[i])
    beta_ra = beta_ra_ + dx[i]
    beta_dec = beta_dec_ + dy[i]
    theta_ra, theta_dec = solver.image_position_from_source(
            beta_ra,
            beta_dec,
            kwargs_interp,
            search_window=np.max(pixel_grid.width),
            min_distance=pixel_grid.pixel_width,
            solver='lenstronomy',
            **kwargs_solver,
        )
    mag_images = lens_model_interp.magnification(theta_ra, theta_dec, kwargs_interp)


    #theta_ra, theta_dec = solver.image_position_from_source(beta_ra, beta_dec, kwargs_interp)
    print ('Lenstronomy:',theta_ra, theta_dec, mag_images)

    #f, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=False, sharey=False)
    #lens_plot.lens_model_plot(ax, lensModel=lens_model_interp, kwargs_lens=kwargs_interp,
    #                          numPix=numPix, deltaPix=deltaPix,
    #                          sourcePos_x=beta_ra, sourcePos_y=beta_dec, point_source=True, with_caustics=True, fast_caustic=True, coord_inverse=False)


    kwargs_psr = {
        'zs': 3.0,
        'ys1': beta_ra,
        'ys2': beta_dec,
        'flux': 1.0
    }


    ps=pointsrc(size=fov_, sizex=None, sizey=None, Npix=numPix, gl=df, **kwargs_psr)
    xi,yi,mui = ps.xi1, ps.xi2, ps.mui
    print ('pyLensLib:',xi,yi,mui)

    #ax.plot(xi,yi,'o',color='red')

    #plt.show()

    ps=pointsrc(size=fov_, sizex=None, sizey=None, Npix=numPix, gl=df, use_lenstronomy=True, **kwargs_psr)
    xi,yi,mui = ps.xi1, ps.xi2, ps.mui
    print ('pyLensLenstr:',xi,yi,mui)







