from pyLensLib.sersic import sersic
from pyLensLib.poststamp import poststamp
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import matplotlib.pyplot as plt
co = FlatLambdaCDM(H0=70.0, Om0=0.3)
import astropy.io.fits as fits
from PIL import Image



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

npix=100
fsize =10.0
sizex = [-5.+5, 5.+5]
sizey = [-5.+5, 5.+5]
se = sersic(Npix=npix, gl=None, save_unlensed=True,
	rmaxf=10.0, sizex=sizex, sizey=sizey, **kwargs)


primary_hdu = fits.PrimaryHDU(se.image)
hdul=fits.HDUList([primary_hdu])
hdul.writeto('test.fits',overwrite=True)

#plt.imshow(np.log10(se.image))
#plt.show()


kwargs = {'px': fsize/(npix-1),'zs': 1.0, 'flux': se.image.sum()}
ps = poststamp(se.image,Npix=npix,size=fsize,gl=None,save_unlensed=True,**kwargs)

primary_hdu = fits.PrimaryHDU(ps.image)
hdul=fits.HDUList([primary_hdu])
hdul.writeto('test2.fits',overwrite=True)


xi=np.arange(npix)+0.5
yi=np.arange(npix)+0.5
x,y=np.meshgrid(xi,xi)

from scipy.ndimage import map_coordinates

im = map_coordinates(se.image,[y,x],order=1)
primary_hdu = fits.PrimaryHDU(im)
hdul=fits.HDUList([primary_hdu])
hdul.writeto('test3.fits',overwrite=True)



"""
background = Image.fromarray(np.zeros((npix*2,npix*2)))
foreground = Image.fromarray(se.image)

x, y = foreground.size
background.paste(foreground,(0,0))
primary_hdu = fits.PrimaryHDU(np.asarray(background))
hdul=fits.HDUList([primary_hdu])
hdul.writeto('test3.fits',overwrite=True)
"""