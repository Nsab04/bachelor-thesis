from pyLensLib.nfwell import nfwell
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

mass = np.logspace(14,15.5,10000)

zl = 0.5
zs = 2.0
c = []
c_nom = []
for i in range(len(mass)):
    kwargs_host = {'zl': zl, 'zs': zs, 'mass': mass[i], 'x1': 0.0, 'x2':0.0}
    nfw=nfwell(co,**kwargs_host)
    c.append(nfw.conc)
    c_nom.append(nfw.conc_nom)

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.plot(mass,c,'+')
ax.plot(mass,c_nom,'-')
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()