from pyLensLib.raytracer import raytracer
from pyLensLib.deflector import deflector
from pyLensLib.nfwell import nfwell
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

kwargs_host = {'zl': 0.5, 'zs': 2.0, 'mass': 1e15, 'conc': 4.0, 'x1': 0.0, 'x2':0.0}
nfw=nfwell(co,**kwargs_host)
theta = np.linspace(-100.0,100,512)
nfw.setGrid(theta=theta)

massmap = (nfw.ka*nfw.sigma_crit()*np.deg2rad(nfw.pixel_scale/3600.0)**2*
           co.angular_diameter_distance(0.5).value**2)


kwargs = {'zl': 0.5, 'zs': 2.0, 'fov': 200}
rayt=raytracer(co,massmap,Nray=512,FOVray=200.0,fromfile=False,**kwargs)


kwargs_def={'zl': 0.5, 'zs': 2.0}
df=deflector(co,pot=rayt.pot,usePotential=True,**kwargs_def)
df.setGrid(theta=theta,compute_potential=True)

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(df.ka,origin='lower',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
plt.show()



