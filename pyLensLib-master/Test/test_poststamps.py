import numpy as np
import matplotlib.pyplot as plt
from pyLensLib.poststamp import poststamp
from pyLensLib.piemd import piemd
from astropy.cosmology import FlatLambdaCDM

# create a test deflector (in this case using a piemd model)
co = FlatLambdaCDM(H0=70.0, Om0=0.3)

kwargs = {'zl': 0.5,
          'zs': 1.0,
          'sigma0': 200.0,
          'q': 0.6,
          'pa': -np.pi / 4.0,
          'theta_c': 0.001,
          'theta_t': 10.0,
          'x1': 0.0,
          'x2': 0.0}

dpie=piemd(co,**kwargs)
theta = np.linspace(-5,5,2048)
dpie.setGrid(thetax=theta,thetay=theta)

# read galaxy images from hudf catalog
import h5py
hf = h5py.File('/Users/maxmen3/stiva/HUDF/hudf_dataset.h5', 'r')
gals=hf.get('hudf_resized')[()]
sizes = hf.get('hudf_size')[()]
z = hf.get('hudf_z')[()]
mags = hf.get('hudf_mags')[()]
sedt = hf.get('hudf_template')[()]


print (gals.shape)
print (mags.shape)
id = np.argmax(sizes)

print (sedt[id],sizes[id],mags[id,:])

# display unlensed galaxy image
plt.imshow(gals[id,2,:,:].T)
plt.show()

# create a postage stamp
kwargs_src = {
    'px': sizes[id]/(gals[id,2,:,:].shape[0]-1),
    'pa': 0.0,
    'ys1': 0.0,
    'ys2': 0.0,
    'flux': 1.0, # at the moment I am setting an arbitrary value
    'zs': z[id]
}
ps = poststamp(gals[id,3,:,:],Npix=2048,size=theta.max()*2.0,**kwargs_src,gl=dpie)

plt.imshow(ps.image,extent=[theta.min(),theta.max(),theta.min(),theta.max()])
dpie.change_redshift(z[id])
tcl=dpie.tancl()
rcl=dpie.radcl()
for c in tcl:
    x,y=dpie.getCritPoints(c)
    plt.plot(x,y,'-',color='w')
for c in rcl:
    x,y=dpie.getCritPoints(c)
    plt.plot(x,y,'-',color='w')
plt.show()



