from pyLensLib.sersic import sersic
from pyLensLib.poststamp import poststamp
import numpy as np

class sersic_sourceplane(object):
    """
    Populate a source plane with sources having Sersic profiles.

    Attributes:
        nsrc (int): Number of sources.
        n, re, q, pa, ys1, ys2, c, flux (array): Source parameters.
        zs (float): Source redshift.
        image_unlensed (ndarray): Combined unlensed image.
        ps (poststamp): Postage stamp object for the unlensed image.
    """
    def __init__(self, size=100.0, Npix=100, x1c=0.0, x2c=0.0, y1c=0.0, y2c=0.0, gl=None,
                 sizex=None, sizey=None, pcx=None, pcy=None,
                 mask=None, save_unlensed=False, rmaxf=100,**srcdict):
        """
        Initialize a sersic source plane instance.

        Args:
            size (float): Size of the image in arcsec.
            Npix (int): Number of pixels.
            x1c, x2c, y1c, y2c (float): Center coordinates for the postage stamp.
            gl: Lens object.
            sizex, sizey (tuple, optional): x/y range for custom frame.
            pcx, pcy (array, optional): Pixel coordinates in x/y.
            mask (ndarray, optional): Mask for image.
            save_unlensed (bool): Save unlensed image if True.
            rmaxf (float): Maximum radius for evaluation.
            srcdict: Dictionary of source parameters (nsrc, n, re, q, pa, ys1, ys2, c, flux, zs).

        Returns:
            None
        """
        if ('nsrc' in srcdict):
            self.nsrc = srcdict['nsrc']
        else:
            raise ValueError("Missing number of sources in srcdict")

        if ('n' in srcdict):
            self.n = srcdict['n']
        else:
            self.n = np.ones(self.nsrc)*4.0

        if ('re' in srcdict):
            self.re = srcdict['re']
        else:
            self.re = np.ones(self.nsrc)*5.0

        if ('q' in srcdict):
            self.q = srcdict['q']
        else:
            self.q = np.ones(self.nsrc)*1.0

        if ('pa' in srcdict):
            self.pa = srcdict['pa']
        else:
            self.pa = np.ones(self.nsrc)*0.0

        if ('ys1' in srcdict):
            self.ys1 = srcdict['ys1']
        else:
            self.ys1 = np.ones(self.nsrc)*0.0

        if ('ys2' in srcdict):
            self.ys2 = srcdict['ys2']
        else:
            self.ys2 = np.ones(self.nsrc)*0.0

        if ('c' in srcdict):
            self.c = srcdict['c']
        else:
            self.c = np.ones(self.nsrc)*0.0

        if ('flux' in srcdict):
            self.flux = srcdict['flux']
        else:
            self.flux = np.ones(self.nsrc)*100.0

        if ('zs' in srcdict):
            self.zs = srcdict['zs']
        else:
            self.zs = 1.0

        for i in range(self.nsrc):
            kwargs = {
                'n': self.n[i],
                're': self.re[i],
                'q': self.q[i],
                'pa': self.pa[i],
                'ys1': self.ys1[i],
                'ys2': self.ys2[i],
                'c': self.c[i],
                'flux': self.flux[i],
                'zs': self.zs
            }
            se = sersic(size=size, Npix=Npix, gl=None,
                 sizex=sizex, sizey=sizey, pcx=pcx, pcy=pcy,
                 mask=mask, save_unlensed=save_unlensed, rmaxf=rmaxf,**kwargs)

            if (i == 0):
                image_unlensed = se.image.copy()
            else:
                image_unlensed = image_unlensed.copy() + se.image.copy()

        self.image_unlensed = image_unlensed

        # create a postage stamp with the unlensed image
        kwargs = {'px': size/(Npix-1),'zs': self.zs, 'flux': image_unlensed.sum(), 'ys1': y1c, 'ys2': y2c}
        self.ps = poststamp(image_unlensed,Npix=Npix,size=size,gl=gl,save_unlensed=save_unlensed,xc=x1c,yc=x2c,**kwargs)
