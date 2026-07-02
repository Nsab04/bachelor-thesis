import numpy as np
from scipy.special import genlaguerre
from pyLensLib.gensrc import gensrc
from itertools import product



class shapelets_polar(gensrc):
    """
    Shapelets source model in polar coordinates for gravitational lensing.

    Attributes:
        rs (float): Scale radius.
        f (dict): Shapelet coefficients.
        nmax (int): Maximum shapelet order.
        ys1, ys2 (float): Source center coordinates.
        q (float): Axis ratio.
        pa (float): Position angle (radians).
        zs (float): Source redshift.
        rescf (float): Rescaling factor for lensing geometry.
        mask (ndarray): Mask for image.
        image, image_unlensed (ndarray): Lensed and unlensed images.
        x1, x2, y1, y2 (ndarray): Meshgrid coordinates.
    """

    def __init__(self, size=100.0, Npix=100, gl=None, sizex=None, sizey=None, pcx=None, pcy=None, mask=None, save_unlensed=False, rmaxf=100, **kwargs):
        """
        Initialize the shapelets source in polar coordinates.

        Args:
            size (float): Size of the image in arcsec.
            Npix (int): Number of pixels.
            gl: Instance of the lens class.
            sizex, sizey (tuple, optional): x/y range for custom frame.
            pcx, pcy (array, optional): Pixel coordinates in x/y.
            mask (ndarray, optional): Mask for image.
            save_unlensed (bool): Save unlensed image if True.
            rmaxf (float): Maximum radius for evaluation.
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
            self.f = kwargs['f']
        else:
            self.f = dict([('fS00', 1)])

        if ('nmax' in kwargs):
            self.nmax = kwargs['nmax']
        else:
            self.nmax = 0

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

        self.image = self.brightness(y1, y2, rmaxf)
        if (save_unlensed):
            self.image_unlensed = self.brightness(self.x1, self.x2, rmaxf)

        if self.mask is not None:
            image_mask[self.mask] = self.image
            self.image = image_mask

    def brightness(self, y1, y2, rmaxf):
        """
        Evaluate the brightness of the source on a grid using shapelets in polar coordinates.

        Args:
            y1, y2 (ndarray): Grid coordinates.
            rmaxf (float): Maximum radius for evaluation.

        Returns:
            ndarray: Brightness map.
        """

        x = np.cos(self.pa) * (y1 - self.ys1) + np.sin(self.pa) * (y2 - self.ys2)
        y = -np.sin(self.pa) * (y1 - self.ys1) + np.cos(self.pa) * (y2 - self.ys2)
        r = np.sqrt((x / self.q) ** 2 + y ** 2) / self.rs
        isel = r < rmaxf * self.rs

        phi = np.asarray(np.arctan2(y, x), dtype=float)

        brightness = np.zeros_like(r)

        for n in np.arange(self.nmax+1):
            for m in np.arange(-n, n+1, 2):
                mn = 'fS'+str(n)+str(m)
                an = mn + 'r'
                if mn in self.f:
                    modulus = self.f[mn]
                    if an in self.f:
                        angle = np.deg2rad(self.f[an])
                    else:
                        angle = 0
                else:
                    modulus = 0
                    angle = 0

                brightness[isel] += modulus * np.cos(-m*phi[isel] + angle) * self.xi(r[isel], n, m)

        return brightness

    def xi(self, r, n, m):
        """
        Compute the radial shapelet basis function xi(r, n, m).

        Args:
            r (float or array): Radius.
            n (int): Shapelet order.
            m (int): Azimuthal order.

        Returns:
            float or array: Value of the shapelet basis function.
        """

        ma = np.abs(m)
        L = genlaguerre(n=(n-ma)/2, alpha=ma)

        coeff1 = -1**(0.5*(n-ma))/(self.rs**(ma+1)) * ((np.math.factorial((n-ma)/2))/(np.pi * np.math.factorial((n+ma)/2)))**0.5
        coeff2 = r**ma * L(np.asarray((r/self.rs)**2))

        return -coeff1 * coeff2 * np.exp(-0.5*(r/self.rs)**2)
