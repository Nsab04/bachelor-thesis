from pyLensLib.subfind import subfind
from pyLensLib.deflector import deflector
from pyLensLib.cluster import cluster
from tqdm import tqdm
import pymupds

import numpy as np
import astropy.io.fits as fits

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyLensLib.lenstool as lst

def showcl(cl,ax):
    for c in cl:
        vs = c.points
        x, y = zip(*vs)
        if (c.principale):
            ax.plot(x, y, '-', color='black')
        else:
            ax.plot(x, y, '-', color='red')

#snapshot = '/Users/massimo/projects/ggsl/testing_codes/massmaps_python/BH_2015/snap_058'
#subfind_file = "/Users/massimo/stiva/Dianoga/new_elena/Subfind_AGN/D1/sub_058.0"

#sf=subfind(subfind_file)
#xc,yc,zc=sf.gpos[0]
#cl = cluster(snapshot=snapshot,parttype=[1],xc=xc,yc=yc,zc=zc)


#filealpha='/Users/massimo/stiva/Dianoga/new_elena/AGN_400_2048_2/lensing/1_z_058alpha.fits'
#hdul=fits.open(filealpha)
#a1=hdul[0].data
#a2=hdul[1].data

#dls=cl.co.angular_diameter_distance_z1z2(cl.zl,kwargs_def['zs'])
#ds=cl.co.angular_diameter_distance(kwargs_def['zs'])
#a1=a1*dls/ds
#a2=a2*dls/ds

#kwargs_def={'zl': cl.zl, 'zs': 3.0}
#df=deflector(cl.co,angx=a1,angy=a2,**kwargs_def)
#theta=np.linspace(-100,100,2048)
#df.setGrid(theta=theta,compute_potential=False)


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

# # cluster nickname (necessari per costruire i nomi dei file da leggere)
# cluster = ['S1063','M0416_B22','M1206pl','PSZ1G311_200',
#            'A370','A2744','M0717','M1149',
#            'M0329','M1931','M2129','R2129']
#
# # posizione delle mappe degli angoli di deflessione
# path_to_angles = ['/Users/massimo/stiva/pietro_models/',
#                   '/Users/massimo/stiva/pietro_models/',
#                   '/Users/massimo/stiva/pietro_models/',
#                   '/Users/massimo/stiva/clusters/PSZ1G311/delens/',
#                   '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
#                   '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
#                   '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
#                   '/Volumes/GoogleDrive/My Drive/Old_Projects/FF/lens_models/from_params/',
#                   '/Users/massimo/stiva/gabriel_models/best_fit/',
#                   '/Users/massimo/stiva/gabriel_models/best_fit/',
#                   '/Users/massimo/stiva/gabriel_models/best_fit/',
#                   '/Users/massimo/stiva/gabriel_models/best_fit/']


cluster = ['S1063','M0416','M1206pl','PSZ1G311_200',
           'A370','A2744_400','M0717','M1149',
           'M0329','M1931','M2129','R2129']

# posizione delle mappe degli angoli di deflessione
path_to_angles = ['/Users/maxmen3/stiva/pietro_models/',
                  '/Users/maxmen3/stiva/pietro_models/',
                  '/Users/maxmen3/stiva/pietro_models/',
                  '/Users/maxmen3/stiva/clusters/PSZ1G311/delens/',
                  '/Users/maxmen3/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/maxmen3/stiva/pietro_models/',
                  '/Users/maxmen3/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/maxmen3/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/maxmen3/stiva/gabriel_models/best_fit/',
                  '/Users/maxmen3/stiva/gabriel_models/best_fit/',
                  '/Users/maxmen3/stiva/gabriel_models/best_fit/',
                  '/Users/maxmen3/stiva/gabriel_models/best_fit/']

icl = 5

zs = 1.0
zl = float(lst.getLensRedshift(path_to_angles[icl] + cluster[icl] + '.par'))
print (('Working with cluster %s at z=%5.3f') %(cluster[icl],zl))
print ('creating deflector...')

df = lst.create_deflector(parfile = path_to_angles[icl] + cluster[icl] + '.par',
                          filex = path_to_angles[icl] + cluster[icl] + '_angx.fits',
                          filey = path_to_angles[icl] + cluster[icl] + '_angy.fits',
                          zl= zl, zs = zs, zsnorm = 1.0, resc_fact = 1.0)

print ('cosmological parameters: H0=%s, Om0=%s, OmL-%s' % (df.co.H0, df.co.Om0, df.co.Ode0))
co = df.co




zsarr=np.logspace(np.log10(zl+0.2),np.log10(7.0),120)

#snapshots = []

mus = np.zeros(df.a1.shape)
snapshots = np.zeros((len(zsarr),df.a1.shape[0],df.a1.shape[1]))
for zs_ in tqdm(zsarr):
    df.change_redshift(zs_)
    detA = (1.0 - df.ka) ** 2 - (df.g1 ** 2 + df.g2 ** 2)
    y1 = df.theta1 - df.a1 + 2 * df.pixel_scale
    y2 = df.theta2 - df.a2 + 2 * df.pixel_scale

    nray = int(df.nray1)  # Ensure 32-bit integer for compatibility

    # Ensure xray, yray, and deta are also Fortran-contiguous arrays
    xray = np.asfortranarray((y1 - df.theta1.min()).reshape(nray, nray), dtype=np.float64)
    yray = np.asfortranarray((y2 - df.theta2.min()).reshape(nray, nray), dtype=np.float64)
    deta = np.asfortranarray(detA.reshape(nray, nray), dtype=np.float64)

    # Ensure deray is a scalar float
    deray = np.float64(df.pixel_scale)  # Explicitly cast to float64
    pymupds.mupds_triangle(mus, nray, xray, yray, deray, deta)
    #snapshots.append(mus.T)
    snapshots[len(snapshots)-len(zsarr)+zsarr.tolist().index(zs_),:,:] = mus.T

print (len(snapshots))
print (f"Snapshots shape: {np.array(snapshots).shape}")
snapshots = np.array(snapshots)

a = snapshots[0,:,:]
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)
im = ax.imshow(a, interpolation='none', aspect='auto',vmax=1.7,vmin=-1.,
               extent=(df.theta1.min(),df.theta1.max(),df.theta2.min(),df.theta2.max()),cmap='viridis')  # , vmin=0, vmax=1)
lambda_text = ax.text(0.75, 0.9, '', transform=ax.transAxes, color='red', fontsize=20)
ax.set_xlabel(r'$\theta_1$ [arcsec]')
ax.set_ylabel(r'$\theta_2$ [arcsec]')


for i in range(len(snapshots)):
    print(f"Frame {i}: min={snapshots[i,:,:].min()}, max={snapshots[i,:,:].max()}")


def animate_func(i):
    #if i % fps == 0:
    #   print( '.', end ='' )
    im.set_array(np.flip(np.log10(snapshots[i,:,:]+1.0), axis=0))
    lambda_text.set_text(r'$z_s = %.2f$' % zsarr[i])
    print(f"Frame {i}: min={snapshots[i,:,:].min()}, max={snapshots[i,:,:].max()}")
    return [im]


print(' ')
fname='mupds_2744'
nSeconds=10
fps = int(float(len(snapshots))/float(nSeconds))
anim = animation.FuncAnimation(fig, animate_func, frames=nSeconds * fps, interval=1000 / fps)
anim.save(fname + '.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])