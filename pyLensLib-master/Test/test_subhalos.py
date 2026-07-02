from pyLensLib.subhaloPop import subhaloPop
from pyLensLib.nfwell import nfwell
from astropy.cosmology import FlatLambdaCDM
co = FlatLambdaCDM(H0=70.0, Om0=0.3)



#kwargs = {
#!    'zhalo': 0.35,
#    'cvir': 5.0
#}
mhalo = 1e15
cvir = 5.0
zhalo=0.5

kwargs = {'zhalo':zhalo,
          'mmin': 1e10,
          'mnorm': mhalo,
          'cvir': cvir,
          'halo_ell2d': 0.33,
}

sh = subhaloPop(**kwargs)

import matplotlib.pyplot as plt

kwargs = {'zl': 0.5,
          'zs': 2.0,
          'mass': mhalo,
          'q': 0.5,
          'conc': cvir,
          'pa': 0.0,
          'x1': 0.0,
          'x2': 0.0}

nfw = nfwell(co, **kwargs)

import numpy as np
print (sh.mr)
print (sh.r)


dl = co.angular_diameter_distance(zhalo)
x = np.rad2deg(sh.x*nfw.R200()/dl.value)*3600
y = np.rad2deg(sh.y*nfw.R200()/dl.value)*3600

mag = sh.Mag+co.distmod(kwargs['zl']).value
flux = 10**(-0.4*(mag - 8.9))
print (flux)

import pandas as pd
data = {'x':x,'y':y,'mag':mag,'flux':flux}
df=pd.DataFrame.from_dict(data=data)


import seaborn as sns
#fig,ax =plt.subplots(1,1,figsize=(10,10))
#ax.plot(x,y,'o')
sns.relplot(x='x',y='y',size='flux',sizes=(40,400),alpha=0.5,palette='muted',data=df)
#ax.set_xlim([-90,90])
#ax.set_ylim([-90,90])
#ax.set_aspect('equal')
plt.show()
