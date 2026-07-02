import numpy as np
from pyLensLib.gensrc import gensrc
from astropy.modeling.functional_models import Sersic2D

class sersic2(gensrc):

    def __init__(self, size=100.0, Npix=100, gl=None, sizex=None, sizey=None, pcx=None, pcy=None, mask=None, save_unlensed=False, rmaxf=100, **kwargs):
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

        if ('Ie' in kwargs):
            self.Ie = kwargs['Ie']
        else:
            self.Ie = 100.0

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
            self.image_unlensed = self.brightness(self.x1, self.x2, rmaxf=rmaxf)

        if self.mask is not None:
            image_mask[self.mask] = self.image
            self.image = image_mask

    def brightness(self, y1, y2, rmaxf):
        px = self.sizex / (self.Nx - 1)

        brightness = np.zeros_like(y1)
        isel = (y1 < rmaxf*self.re + self.ys1) & (y2 < rmaxf*self.re + self.ys2)

        s = Sersic2D(amplitude=self.Ie, r_eff=self.re, n=self.n, x_0=self.ys1, y_0=self.ys2,
                     ellip=1-self.q, theta=self.pa + np.pi/2)

        brightness[isel] = s(y1[isel], y2[isel])*px*px

        return brightness
