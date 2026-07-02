import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from pyLensLib.maputils import map_obj, contour_fit, image_fit
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import cv2

filemap = '/Users/maxmen3/projects/ggsl/map_058_1_2_sph.fits'
hdul=fits.open(filemap)
mappa=hdul[0].data


cosmo = FlatLambdaCDM(Om0=0.24,H0=72)
DL = cosmo.angular_diameter_distance(hdul[0].header['ZL'])
conv = (np.pi / 180.0 / 3600.0 * DL.value)
dx = hdul[0].header['CDELT2'] * 3600.0
mappa = mappa/dx/dx
m=map_obj(mappa)
rbin,r,p,psigma=m.radial_profile(nbins=30,logspace=True,rmin=100,rmax=500,auto_center=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].plot(r*dx, p, '-o')
ax[0].errorbar(r*dx,p,(p-psigma,p+psigma))
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlim([np.min(r)*dx, np.max(r)*dx])
ax[0].set_xlabel('r [arcsec]')
ax[0].set_ylabel(r'$\Sigma(r)$ [$M_\odot$ arcsec$^{-2}$]')
ax[0].set_aspect('equal')
ax[1].imshow(np.sqrt(mappa), cmap='magma',origin='lower')
ax[1].imshow(rbin, alpha=0.2, cmap=plt.cm.nipy_spectral,origin='lower')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.imshow(np.sqrt(mappa), cmap='gray_r', origin='lower')

thr_ = []
ell_ = []
phi_ = []

centr_shift_ = []

for f in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35,  0.5, 0.7, 0.9]:
    cnt = m.get_contours(lev=np.max(mappa) * f)
    print(cnt[0].area)
    for c in cnt:
        c_fit = contour_fit(c)
        if (len(c_fit.x_list)>15):
            #print (len(c_fit.x_list))
            c_fit.fitEllipse()
            center = c_fit.ellipse_center()
            phi = c_fit.ellipse_angle_of_rotation()
            axes = c_fit.ellipse_axis_length()
            ax.plot(c_fit.x_list,c_fit.y_list,'-',color='black')
            R = np.arange(0, 2.0*np.pi, 0.01)
            a, b = axes
            xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
            yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
            ax.plot(xx, yy, '--', color='red')

import pandas as pd
from skimage.measure import EllipseModel
#import spatial_efd
for f in np.logspace(-1,0,20):#[0.1, 0.15, 0.2, 0.25, 0.3, 0.35,  0.5, 0.7, 0.9]:
    cnt = m.get_contours(lev=np.max(mappa)*0.9 * f)

    if len(cnt) == 0:
        continue

    for c in [cnt[0]]:
        #x_,y_,centroid = spatial_efd.ProcessGeometry(cnt[0].geometria)
        #print ('from spatial_efd:', x_,y_,centroid)
        c_fit = contour_fit(c)
        #d = {'x': c_fit.x_list, 'y': c_fit.y_list}
        #df = pd.DataFrame.from_dict(d)
        #df.to_csv('test_contour.csv')
        if (len(c_fit.x_list)>15):
            #print (len(c_fit.x_list))
            c_fit.fitEllipse()
            center = c_fit.ellipse_center()
            phi = c_fit.ellipse_angle_of_rotation()
            axes = c_fit.ellipse_axis_length()
            ax.plot(c_fit.x_list,c_fit.y_list,'-',color='black')
            R = np.arange(0, 2.0*np.pi, 0.01)
            a, b = axes
            xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
            yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
            ax.plot(xx, yy, '-', color='orange')
            thr_.append(np.sqrt(c.area/np.pi))
            ell_.append(abs((a-b)/(a+b)))
            centr_shift_.append(np.sqrt((center[0]-mappa.shape[0]/2.)**2
                                       +(center[1]-mappa.shape[1]/2.)**2))
            phi_.append(phi/np.pi*180.0)


            #### Alternative method using skimage.measure.EllipseModel
            ellipse = EllipseModel()
            estimated = ellipse.estimate(c.points)
            if estimated:
                print ('-----------------')
                print (ellipse.params)
                print (center,a,b,phi)


                a = ellipse.params[3]
                b = ellipse.params[2]
                center = [ellipse.params[0],ellipse.params[1]]
                phi = ellipse.params[4] + np.pi/2
                xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
                yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
                ax.plot(xx, yy, '-', color='green')


axins = inset_axes(ax, width=2.5, height=1.6)
axins.plot(thr_,ell_,'-')
ax.set_xlim([0,2048])
ax.set_ylim([0,2048])



plt.show()
