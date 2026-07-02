import numpy as np
from scipy.special import hermite
from pyLensLib.gensrc import gensrc
from itertools import product


class shapelets(gensrc):
    """
    Shapelets source model in Cartesian coordinates for gravitational lensing.

    Attributes:
        rs (float): Scale radius.
        f (ndarray): Shapelet coefficients.
        ys1, ys2 (float): Source center coordinates.
        q (float): Axis ratio.
        pa (float): Position angle (radians).
        zs (float): Source redshift.
        rescf (float): Rescaling factor for lensing geometry.
        mask (ndarray): Mask for image.
        image, image_unlensed (ndarray): Lensed and unlensed images.
        x1, x2, y1, y2 (ndarray): Meshgrid coordinates.
    """

    def __init__(self, size=100.0, Npix=100, gl=None, sizex=None, sizey=None, pcx=None, pcy=None, mask=None, save_unlensed=False, **kwargs):
        """
        Initialize the shapelets source in Cartesian coordinates.

        Args:
            size (float): Size of the image in arcsec.
            Npix (int): Number of pixels.
            gl: Instance of the lens class.
            sizex, sizey (tuple, optional): x/y range for custom frame.
            pcx, pcy (array, optional): Pixel coordinates in x/y.
            mask (ndarray, optional): Mask for image.
            save_unlensed (bool): Save unlensed image if True.
            **kwargs: Shapelet and source parameters.

        Returns:
            None
        """
        super().__init__()
        if ('rs' in kwargs):
            self.rs = kwargs['rs']
        else:
            self.rs = 1

        if ('f' in kwargs):
            # This should be a list of the type [[f11,f12,...],[f21,f22,...],...]
            self.f = np.asarray(kwargs['f'], dtype=float)
        else:
            self.f = np.asarray([1], dtype=float)

        if ('ys1' in kwargs):
            self.ys1 = kwargs['ys1']
        else:
            self.ys1 = 0.0

        if ('ys2' in kwargs):
            self.ys2 = kwargs['ys2']
        else:
            self.ys2 = 0.0

        if ('q' in kwargs):
            self.q = kwargs['q']
        else:
            self.q = 1.0

        if ('pa' in kwargs):
            self.pa = kwargs['pa']
        else:
            self.pa = 0.0

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

        self.mask = mask
        self.save_unlensed = save_unlensed

        # define the pixel coordinates
        if pcx is not None or pcy is not None:
            self.Nx = len(pcx)
            self.Ny = len(pcx)
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

        self.image = self.brightness(y1, y2)
        if (save_unlensed):
            self.image_unlensed = self.brightness(self.x1, self.x2)

        if self.mask is not None:
            image_mask[self.mask] = self.image
            self.image = image_mask

    def brightness(self, y1, y2):
        """
        Evaluate the brightness of the source on a grid using shapelets in Cartesian coordinates.

        Args:
            y1, y2 (ndarray): Grid coordinates.

        Returns:
            ndarray: Brightness map.
        """
        ff = self.f.flatten()
        brightness = np.zeros_like(y1)

        ncomb = product(np.arange(len(self.f)), repeat=2)
        for fn, nn in zip(ff, ncomb):
            if fn != 0:
                brightness += fn * self.phi_n(y1, y2, n=nn) / (self.rs * (1 + self.q))
        return brightness

    def phi_n(self, y1, y2, n=(1, 1)):
        """
        Compute the shapelet basis function phi_n(y1, y2, n).

        Args:
            y1, y2 (ndarray): Grid coordinates.
            n (tuple): Shapelet order (n1, n2).

        Returns:
            ndarray: Value of the shapelet basis function.
        """
        H1 = hermite(n[0])
        H2 = hermite(n[1])
        nt = n[0] + n[1]

        x = 1/self.rs * (np.cos(self.pa) * (y1 - self.ys1) + np.sin(self.pa) * (y2 - self.ys2))
        y = 1/(self.rs * self.q) * (-np.sin(self.pa) * (y1 - self.ys1) + np.cos(self.pa) * (y2 - self.ys2))

        # z1 = 1/rs * (y1 * np.cos(pa) + y2 * np.sin(pa))
        # z2 = 1/(rs * q) * (y2 * np.cos(pa) - y1 * np.sin(pa))

        return np.asarray(H1(x) * H2(y) * np.exp(-0.5*(x**2 + y**2)) * (2**nt * np.pi * np.math.factorial(n[0]) * np.math.factorial(n[1]))**-0.5, dtype=float)
