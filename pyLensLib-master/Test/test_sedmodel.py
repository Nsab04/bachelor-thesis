from pyLensLib.sedmodel import clusterSEDs
import matplotlib.pyplot as plt


kwargs = {
    'seddir': '/Users/massimo/CODES/bpz-1.99.3/SED/',
    'sed_list': 'eB11.list',
    'morphology': ['ellptical','elliptical','spiral','elliptical'],
    'mag_ref': [20.0, 18.0, 23.0, 21.0],
    'templates_ell': [0],
    'templates_sp': [5,6],
    'zhalo': 0.5,
    'reference_band': '/Users/massimo/CODES/bpz-1.99.3/FILTER/HST_ACS_WFC_F814W.res'
}

cls = clusterSEDs(**kwargs)
fig,ax = plt.subplots(1,1,figsize=(10,10))
for w, fl in zip(cls.waz, cls.flz):
    ax.plot(w,fl,'-')

ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

