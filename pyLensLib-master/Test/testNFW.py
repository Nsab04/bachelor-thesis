from pyLensLib.nfwell import nfwell
import numpy as np
from astropy.cosmology import FlatLambdaCDM

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

def narcs(theta_e):
    return 0.81+0.03*theta_e**1.54

mass = np.logspace(14,15,30)
conc = np.linspace(7.0,4.0,30)
m=[]

theta_e=[]
for i in range(len(mass)):
    kwargs = {'zl': 0.5,
            'zs': 2.0,
            'mass': mass[i],
            'q': 0.7,
            'conc': conc[i],
            'pa': 0.0,
            'x1': 0.0,
            'x2': 0.0}

    p = nfwell(co, **kwargs)
    theta = np.linspace(-100, 100, 512)
    p.setGrid(theta)
    cl = p.tancl()
    if (len(cl)>0):
        theta_e_=np.sqrt(cl[0].area/np.pi)*200.0/len(theta)
        na=narcs(theta_e_)
        print ('%12.4e %8.4f %8.4f' % (mass[i],theta_e_,na*0.25))
        m.append(mass[i])
        theta_e.append(theta_e)

print (len(m),len(theta_e))
#fig,ax=plt.subplots(1,1,figsize=(10,10))
#ax.plot(m,theta_e,'-')
#ax.set_xscale('log')
#fig.savefig('theta_e_mass.png')