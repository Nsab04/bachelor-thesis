from numba import config, jit, njit, prange
config.NUMBA_THREADING_LAYER_PRIORITY="omp tbb workqueue"
config.THREADING_LAYER = 'threadsafe'
from scipy.special import gammaln
from pyLensLib.gensrc import gensrc
import numpy as np
from math import lgamma
import os

os.environ['OMP_DISPLAY_ENV'] = 'FALSE'

'''
Optimized version of the sersic class using numba. The following functions are numba optimized:
'''
@jit(nopython=True, cache=True)
def beta(z, w):
    """
    Compute the beta function using gamma functions.

    Args:
        z (float): First argument.
        w (float): Second argument.

    Returns:
        float: Beta function value.
    """
    return np.exp(lgamma(z) + lgamma(w) - lgamma(z + w))

# @jit(nopython=True, parallel=True)
# def evalbrightness(px, y1, y2, ys1, ys2, pa, q, sie, re, bn, n):
#     inv_n = 1.0 / n
#     x = np.cos(pa) * (y1 - ys1) + np.sin(pa) * (y2 - ys2)
#     y = -np.sin(pa) * (y1 - ys1) + np.cos(pa) * (y2 - ys2)
#     r = np.hypot(x / q, y)
#     brightness = sie * np.exp(-bn * (np.power(r / re, inv_n) - 1.0)) * px * px
#     return brightness

@njit(parallel=True, cache=True, fastmath=True)
def evalbrightness(y1, y2, ys1, ys2, cos_pa, sin_pa, inv_q, inv_re, inv_n,
                   amp, bn, rmax):
    """
    Evaluate the Sersic brightness profile on a 2D grid.

    Args:
        y1, y2 (ndarray): Grid coordinates.
        ys1, ys2 (float): Source center coordinates.
        cos_pa, sin_pa (float): Cached trigonometric position angle values.
        inv_q, inv_re, inv_n (float): Cached reciprocal parameters.
        amp (float): Surface brightness at effective radius times pixel area.
        bn (float): Sersic constant.
        rmax (float): Elliptical-radius cutoff. Use <= 0 to disable.

    Returns:
        ndarray: Brightness map.
    """
    brightness = np.empty_like(y1)
    y1_flat = y1.ravel()
    y2_flat = y2.ravel()
    brightness_flat = brightness.ravel()

    for i in prange(y1_flat.size):
        dx = y1_flat[i] - ys1
        dy = y2_flat[i] - ys2
        x = cos_pa * dx + sin_pa * dy
        y = -sin_pa * dx + cos_pa * dy
        r = np.hypot(x * inv_q, y)

        if rmax > 0.0 and r >= rmax:
            brightness_flat[i] = 0.0
        else:
            brightness_flat[i] = amp * np.exp(-bn * (np.power(r * inv_re, inv_n) - 1.0))

    return brightness

@jit(nopython=True, cache=True, fastmath=True)
def evalbrightness_profile(r, sie, inv_re, inv_n, bn):
    """
    Evaluate the Sersic brightness profile at a given radius.

    Args:
        r (float or array): Radius.
        sie (float): Surface brightness at effective radius.
        inv_re (float): Reciprocal effective radius.
        inv_n (float): Reciprocal Sersic index.
        bn (float): Sersic constant.

    Returns:
        float or array: Brightness at radius r.
    """
    return sie * np.exp(-bn * (np.power(r * inv_re, inv_n) - 1.0))


'''
The sersic class is defined as follows:
'''

class sersic(gensrc):

    """
    Optimized Sersic source class using numba for fast evaluation.

    Attributes:
        n (float): Sersic index.
        re (float): Effective radius.
        q (float): Axis ratio.
        pa (float): Position angle.
        ys1, ys2 (float): Source center coordinates.
        c (float): Sersic profile parameter.
        flux (float): Source flux.
        zs (float): Source redshift.
        rescf (float): Rescaling factor for lensing geometry.
        bn (float): Sersic constant.
        mask (ndarray): Mask for image.
        image, image_unlensed (ndarray): Lensed and unlensed images.
        x1, x2, y1, y2 (ndarray): Meshgrid coordinates.
    """
    def __init__(self, size=100.0, Npix=100, gl=None, sizex=None, sizey=None,
                 pcx=None, pcy=None, mask=None, save_unlensed=False, size_unlensed=100,
                 y1_unlensed=0.0, y2_unlensed=0.0, npix_unlensed=100, save_unlensed_recenter=False, rmaxf=100,
                 **kwargs):
        """
        Set up the Sersic source for lensing simulation.

        Args:
            size (float): Size of the image in arcsec.
            Npix (int): Number of pixels.
            gl: Instance of the lens class.
            sizex, sizey (tuple, optional): x/y range for custom frame.
            pcx, pcy (array, optional): Pixel coordinates in x/y.
            mask (ndarray, optional): Mask for image.
            save_unlensed (bool): Save unlensed image if True.
            size_unlensed (float): Size of the unlensed image.
            y1_unlensed, y2_unlensed (float): Center of unlensed image.
            npix_unlensed (int): Number of pixels in unlensed image.
            save_unlensed_recenter (bool): Recenter unlensed image if True.
            rmaxf (float): Maximum radius (for compatibility).
            **kwargs: Sersic and source parameters.

        Returns:
            None
        """
        super().__init__()
        if ('n' in kwargs):
            self.n = kwargs['n']
        else:
            self.n = 4

        if ('re' in kwargs):
            self.re = kwargs['re']
        else:
            self.re = 5.0

        if ('q' in kwargs):
            self.q = kwargs['q']
        else:
            self.q = 1.0

        if ('pa' in kwargs):
            self.pa = kwargs['pa']
        else:
            self.pa = 0.0

        if ('ys1' in kwargs):
            self.ys1 = kwargs['ys1']
        else:
            self.ys1 = 0.0

        if ('ys2' in kwargs):
            self.ys2 = kwargs['ys2']
        else:
            self.ys2 = 0.0

        if ('c' in kwargs):
            self.c = kwargs['c']
        else:
            self.c = 0.0

        if ('flux' in kwargs):
            self.flux = kwargs['flux']
        else:
            self.flux = 100.0

        if ('zs' in kwargs):
            self.zs = kwargs['zs']
        else:
            self.zs = 1.0


        if gl != None:
            if self.zs != gl.zs:
                if self.zs > gl.zl:
                    ds = gl.co.angular_diameter_distance(self.zs).value
                    dls = gl.co.angular_diameter_distance_z1z2(gl.zl,self.zs).value
                    self.rescf=dls/ds*gl.ds/gl.dls
                else:
                    self.rescf = 0.0
            else:
                self.rescf=1.0
        else:
            self.rescf=1.0

        self.df = gl
        self.bn = 1.992 * self.n - 0.3271

        self.mask = mask
        self.save_unlensed = save_unlensed
        self._brightness_cache_key = None

        # define the pixel coordinates
        if pcx is not None or pcy is not None:
            self.Nx = len(pcx)
            self.Ny = len(pcy)
            self.sizex = np.max(pcx)-np.min(pcx)
            self.sizey = np.max(pcy)-np.min(pcy)
            if np.round((pcx[1]-pcx[0]),6) != np.round((pcy[1]-pcy[0]),6):
                raise Exception('Pixels of pcx and pcy must have the same dimension')

        elif sizex is not None or sizey is not None:
            self.N = Npix
            pcx = np.linspace(sizex[0], sizex[1], self.N)
            pcy = np.linspace(sizey[0], sizey[1], self.N)
            self.Nx = len(pcx)
            self.Ny = len(pcx)
            self.sizex = np.max(pcx)-np.min(pcx)
            self.sizey = np.max(pcy)-np.min(pcy)

        else:
            self.size = size
            self.N = Npix
            pcx = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
            pcy = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
            self.Nx = Npix
            self.Ny = Npix
            self.sizex = float(size)
            self.sizey = float(size)

        self._ensure_brightness_cache()

        self.x1, self.x2 = np.meshgrid(pcx, pcy)

        if mask is not None:
            image_mask = np.full(self.x1.shape,np.nan)
            self.x1, self.x2 = self.x1[self.mask], self.x2[self.mask]

        if self.df != None:
            y1, y2 = self.ray_trace()
        else:
            y1, y2 = self.x1, self.x2

        self.y1 = y1
        self.y2 = y2

        self.image = self.brightness(y1, y2, rmaxf)
        if (save_unlensed):

            if save_unlensed_recenter:
                x1_ = np.linspace(y1_unlensed - size_unlensed/2., y1_unlensed + size_unlensed/2., npix_unlensed)
                x2_ = np.linspace(y2_unlensed - size_unlensed/2., y2_unlensed + size_unlensed/2., npix_unlensed)

                self.x1_, self.x2_ = np.meshgrid(x1_, x2_)
                px_unlensed = size_unlensed / (npix_unlensed - 1)
                self.image_unlensed = self.brightness(
                    self.x1_,
                    self.x2_,
                    rmaxf=rmaxf,
                    pixel_area=px_unlensed * px_unlensed,
                )
            else:
                self.image_unlensed = self.brightness(self.x1, self.x2, rmaxf=rmaxf)

        if self.mask is not None:
            image_mask[self.mask] = self.image
            self.image = image_mask

    def _compute_sie(self):
        return self.flux/(2.0*np.pi*self.re**2*np.exp(self.bn)*self.n*
                          self.bn**(-2.0*self.n)*np.exp(gammaln(2.0*self.n))*self.q)*self.rfun()

    def _ensure_brightness_cache(self):
        px = self.sizex / (self.Nx - 1)
        cache_key = (
            float(self.ys1),
            float(self.ys2),
            float(self.pa),
            float(self.q),
            float(self.re),
            float(self.n),
            float(self.bn),
            float(self.flux),
            float(self.c),
            float(px),
        )

        if cache_key == self._brightness_cache_key:
            return

        self._px = px
        self._pixel_area = px * px
        self._cos_pa = np.cos(self.pa)
        self._sin_pa = np.sin(self.pa)
        self._inv_q = 1.0 / self.q
        self._inv_re = 1.0 / self.re
        self._inv_n = 1.0 / self.n
        self._sie = self._compute_sie()
        self._amp = self._sie * self._pixel_area
        self._brightness_cache_key = cache_key

    def render_on_grid(self, y1, y2, rmaxf=100, pixel_area=None):
        """
        Render the Sersic profile on caller-supplied source-plane coordinates.

        Reusing an already ray-traced grid avoids rebuilding meshgrids and
        repeating interpolation work when many sources share the same frame.
        """
        self._ensure_brightness_cache()
        amp = self._amp if pixel_area is None else self._sie * pixel_area
        rmax = rmaxf * self.re if rmaxf is not None and rmaxf > 0 else -1.0
        return evalbrightness(y1, y2, self.ys1, self.ys2, self._cos_pa, self._sin_pa,
                              self._inv_q, self._inv_re, self._inv_n, amp, self.bn, rmax)

    def brightness(self, y1, y2, rmaxf, pixel_area=None):
        """
        Evaluate the brightness of the source on a grid.

        Args:
            y1, y2 (ndarray): Grid coordinates.
            rmaxf (float): Maximum radius (for compatibility).
            pixel_area (float, optional): Pixel area for caller-supplied grids.
                If omitted, use the primary image grid pixel area.

        Returns:
            ndarray: Brightness map.
        """
        return self.render_on_grid(y1, y2, rmaxf=rmaxf, pixel_area=pixel_area)

    def brightness_profile(self,r):
        """
        Evaluate the brightness of the source at a given radius.

        Args:
            r (float or array): Radius.

        Returns:
            float or array: Brightness at radius r.
        """
        self._ensure_brightness_cache()
        return evalbrightness_profile(r, self._sie, self._inv_re, self._inv_n, self.bn)

    def sie(self):
        """
        Evaluate the surface brightness at the effective radius.

        Returns:
            float: Surface brightness at effective radius.
        """
        self._ensure_brightness_cache()
        return self._sie
    def rfun(self):
        """
        Evaluate the r function for the Sersic profile.

        Returns:
            float: Value of the r function.
        """
        return np.pi*(self.c+2.0)/4.0/beta(1./(self.c+2.0),1.+1./(self.c+2.0))
