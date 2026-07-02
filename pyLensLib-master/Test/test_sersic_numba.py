from pyLensLib.sersic_numba import sersic as sersic_numba
from pyLensLib.sersic import sersic as sersic_standard
import numpy as np
import timeit
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# set up source
kwargs = {
    'n': 1.0,
    'q': 0.8,
    'ys1': 5.0,  # ys1+dy1,
    'ys2': 5.0,  # ys2+dy2,
    'pa': np.pi/7.0,
    're': 0.5,
    'flux': 0.00001,
    'zs': 1.0
}

npix=1000
fsize =10.0
sizex = [-5.+5, 5.+5]
sizey = [-5.+5, 5.+5]

# evaluate time to execute the sersic function and print it
print ('Computing time for sersic_numba and sersic_standard')
print(timeit.timeit('se = sersic_numba(Npix=npix, gl=None, save_unlensed=True, rmaxf=20.0, sizex=sizex, sizey=sizey, **kwargs)', globals=globals(), number=1000))
print(timeit.timeit('se = sersic_standard(Npix=npix, gl=None, save_unlensed=True, rmaxf=20.0, sizex=sizex, sizey=sizey, **kwargs)', globals=globals(), number=1000))

print ('Image statistics for sersic_numba and sersic_standard')
se = sersic_numba(Npix=npix, gl=None, save_unlensed=True, rmaxf=10.0, sizex=sizex, sizey=sizey, **kwargs)
print (se.image.min(), se.image.max(), se.image.mean(), se.image.std())

se = sersic_standard(Npix=npix, gl=None, save_unlensed=True, rmaxf=10.0, sizex=sizex, sizey=sizey, **kwargs)
print (se.image.min(), se.image.max(), se.image.mean(), se.image.std())