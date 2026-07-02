import numpy as np
from scipy.ndimage import map_coordinates
from pyLensLib.gensrc import gensrc

class poststamp(gensrc):

    """
    Class representing a postage stamp image for lensing simulations.

    Attributes:
        px (float): Input pixel scale.
        pa (float): Position angle.
        ys1, ys2 (float): Source position coordinates.
        flux (float): Source flux.
        zs (float): Source redshift.
        rescf (float): Rescaling factor for lensing geometry.
        N (int): Number of pixels in output image.
        size (float): Size of output image.
        df: Deflector object.
        image_in (ndarray): Input image array.
        size_in (float): Size of input image.
        save_unlensed (bool): Save unlensed image if True.
        x1, x2 (ndarray): Meshgrid coordinates.
        px_out (float): Output pixel scale.
        y1, y2 (ndarray): Ray-traced coordinates.
        image (ndarray): Lensed image.
        image_unlensed (ndarray): Unlensed image.
    """
    def __init__(self, image_in, xc=0.0, yc=0.0, size=100.0, Npix=100, gl=None, save_unlensed=False, **kwargs):
        """
        Initialize a postage stamp image for lensing simulation.

        Args:
            image_in (ndarray): Input image array.
            xc, yc (float): Center coordinates for output image.
            size (float): Size of output image.
            Npix (int): Number of pixels in output image.
            gl: Deflector object.
            save_unlensed (bool): Save unlensed image if True.
            **kwargs: Source and lens parameters (px, pa, ys1, ys2, flux, zs).

        Returns:
            None
        """
        super().__init__()
        if ('px' in kwargs):
            self.px = kwargs['px']
        else:
            self.px = 0.03

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
                    self.rescf = dls/ds*gl.ds/gl.dls
                else:
                    self.rescf = 0.0
            else:
                self.rescf = 1.0
        else:
            self.rescf=1.0

        self.N = Npix # number of pixels in the output image
        self.size = float(size) # size of the output image
        self.df = gl
        self.image_in=image_in-image_in.min()
        self.image_in=self.image_in/self.image_in.sum()*self.flux
        self.size_in=self.image_in.shape[0]*self.px
        self.save_unlensed = save_unlensed

        # define the pixel coordinates
        pc = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
        self.px_out=pc[1]-pc[0]
        self.x1, self.x2 = np.meshgrid(pc+xc, pc+yc)
        if self.df != None:
            y1, y2 = self.ray_trace()
        else:
            y1, y2 = self.x1, self.x2

        self.y1, self.y2 = y1, y2
        self.image = self.brightness(y1, y2)
        if (save_unlensed):
            self.image_unlensed = self.brightness(self.x1, self.x2)

    def brightness(self, y1, y2):
        """
        Generate a brightness map for the postage stamp at given coordinates.

        Args:
            y1, y2 (ndarray): Coordinates for brightness evaluation.

        Returns:
            ndarray: Brightness map.
        """
        x = np.cos(self.pa) * (y1 - self.ys1) + np.sin(self.pa) * (y2 - self.ys2)
        y = -np.sin(self.pa) * (y1 - self.ys1) + np.cos(self.pa) * (y2 - self.ys2)
        x1pix = (x + self.size_in / 2.0) / self.px-0.5
        x2pix = (y + self.size_in / 2.0) / self.px-0.5
        brightness = map_coordinates(self.image_in, [x2pix, x1pix], order=1, prefilter=True, mode='constant')
        brightness[brightness < 0.0] = 0.0
        return (brightness/self.px**2*self.px_out**2)