import shapely.geometry
from shapely.ops import polygonize, unary_union
from shapely.geometry import Point, Polygon
from  pyLensLib.critcau import CriticalLine, Caustic
import numpy as np
from scipy.ndimage import map_coordinates
from skimage import measure
import scipy.fftpack as fftengine
from astropy import units as u
from astropy.constants import c
from scipy.spatial import ConvexHull
from matplotlib.path import Path
try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - optional dependency
    gpd = None
import pandas as pd
import matplotlib.pyplot as plt


def apply_window_function(kappa):
    """
    Apply a window function to the convergence map to reduce edge effects.

    Args:
        kappa (np.ndarray): Convergence map.

    Returns:
        np.ndarray: Windowed convergence map.
    """
    window = np.hanning(kappa.shape[0])[:, None] * np.hanning(kappa.shape[1])[None, :]
    return kappa * window

class genlen(object):
    """
    General lensing class for handling lens maps, caustics, and critical lines.

    Attributes:
        initialized (bool): Indicates if the object is initialized.
        theta1, theta2 (np.ndarray): Meshgrid coordinates.
        thetax, thetay (np.ndarray): Axis coordinates.
        mask (np.ndarray): Optional mask for the grid.
        perturbed (bool): Indicates if the lens is perturbed.
        g1, g2 (np.ndarray): Shear components.
        ka (np.ndarray): Convergence map.
        a1, a2 (np.ndarray): Deflection angles.
        F1, F2, G1, G2 (np.ndarray): Flexion maps (if computed).
    """
    def __init__(self):
        """
        Initialize a genlen object.
        """
        self.initialized = True

    def setGrid0(self, theta=None, thetax=None, thetay=None, compute_potential=False, mask=None, computeFlexion=False):
        """
        Set up the grid covering the source plane and compute lens maps such as convergence, shear, and deflection angles.

        Args:
            theta (np.ndarray, optional): Pixel positions along one axis.
            thetax (np.ndarray, optional): X-axis pixel positions.
            thetay (np.ndarray, optional): Y-axis pixel positions.
            compute_potential (bool, optional): If True, compute the potential map (needed for time delay surfaces).
            mask (np.ndarray, optional): Mask to apply to the grid.
            computeFlexion (bool, optional): If True, compute flexion maps.

        Returns:
            None. The computed maps are stored as object properties (e.g., ka, g1, g2, a1, a2, potential).
        """

        # construct a mesh:
        if thetax is None or thetay is None:
            self.theta1, self.theta2 = np.meshgrid(theta, theta)
            self.thetax = theta
            self.thetay = theta
        else:
            self.theta1, self.theta2 = np.meshgrid(thetax, thetay)
            if np.round(thetax[1]-thetax[0], 6) == np.round(thetay[1]-thetay[0], 6):
                self.thetax = thetax
                self.thetay = thetay
            else:
                raise Exception('thetax and thetay must have the same pixel scale '
                                + str(thetax[1]-thetax[0])
                                + ', ' + str(thetay[1]-thetay[0]))


        self.mask = mask
        self.perturbed = False

        if self.mask is not None:
            self.a1 = np.full(self.theta1.shape, np.nan)
            self.a2 = np.full(self.theta1.shape, np.nan)
            self.ka = np.full(self.theta1.shape, np.nan)
            self.g1 = np.full(self.theta1.shape, np.nan)
            self.g2 = np.full(self.theta1.shape, np.nan)

            self.theta1 = self.theta1[self.mask]
            self.theta2 = self.theta2[self.mask]
            self.g1[self.mask], self.g2[self.mask] = self.gamma(self.theta1, self.theta2)
            self.ka[self.mask] = self.kappa(self.theta1, self.theta2)
            self.a1[self.mask], self.a2[self.mask] = self.angle(self.theta1, self.theta2)
            if computeFlexion:
                g11 = np.full(self.theta1.shape, np.nan)
                g22 = np.full(self.theta1.shape, np.nan)
                g12 = np.full(self.theta1.shape, np.nan)
                g21 = np.full(self.theta1.shape, np.nan)
                self.F1 = np.full(self.theta1.shape, np.nan)
                self.F2 = np.full(self.theta1.shape, np.nan)
                self.G1 = np.full(self.theta1.shape, np.nan)
                self.G2 = np.full(self.theta1.shape, np.nan)
                g12[self.mask], g11[self.mask] = np.gradient(self.g1)
                g22[self.mask], g21[self.mask] = np.gradient(self.g2)
                self.F1[self.mask], self.F2[self.mask] = g11 + g22, g21 - g12
                self.G1[self.mask], self.G2[self.mask] = g11 - g22, g21 + g12
        else:
            # self.grid_pixel=self.thetax[1]-self.thetax[0]

            self.g1, self.g2 = self.gamma(self.theta1, self.theta2)
            self.ka = self.kappa(self.theta1, self.theta2)
            self.a1,self.a2=self.angle(self.theta1, self.theta2)
            if computeFlexion:
                g12, g11 = np.gradient(self.g1)
                g22, g21 = np.gradient(self.g2)
                self.F1, self.F2 = g11 + g22, g21 - g12
                self.G1, self.G2 = g11 - g22, g21 + g12

        self.size1 = np.max(self.thetax) - np.min(self.thetax)
        self.size2 = np.max(self.thetay) - np.min(self.thetay)
        self.nray1 = len(self.thetax)
        self.nray2 = len(self.thetay)

        self.pixel_scale = self.thetax[1]-self.thetax[0]

        if self.computed_potential:
            self.conv_fact_time = \
                ((1. + self.zl) / c.to(u.km / u.s) *
                 (self.dl * self.ds / self.dls).to(u.km)).to(u.d) * \
                (np.pi / 180.0 / 3600.) ** 2

        # self.computed_potential=False
        if not self.computed_potential and compute_potential:
            self.potential_from_kappa()
            self.computed_potential = True
            self.conv_fact_time = \
                ((1. + self.zl) / c.to(u.km / u.s) *
                 (self.dl * self.ds / self.dls).to(u.km)).to(u.d) * \
                (np.pi / 180.0 / 3600.) ** 2

    def arcsec2pixel(self, x_arcsec, y_arcsec):
        """
        Convert arcsecond coordinates to pixel coordinates based on the grid defined by thetax and thetay.

        Args:
            x_arcsec (float or np.ndarray): X coordinate(s) in arcseconds.
            y_arcsec (float or np.ndarray): Y coordinate(s) in arcseconds.

        Returns:
            tuple: (x_pixel, y_pixel) coordinates in pixels.
        """
        x_pixel = (x_arcsec - self.thetax[0]) / self.pixel_scale
        y_pixel = (y_arcsec - self.thetay[0]) / self.pixel_scale
        return x_pixel, y_pixel

    def setGrid(self, theta=None, thetax=None, thetay=None, compute_potential=False, mask=None, computeFlexion=False):
        """
        Set up the grid and compute lens maps (convergence, shear, deflection angles).

        Args:
            theta (np.ndarray, optional): Pixel positions along one axis.
            thetax (np.ndarray, optional): X-axis pixel positions.
            thetay (np.ndarray, optional): Y-axis pixel positions.
            compute_potential (bool, optional): If True, compute the potential map.
            mask (np.ndarray, optional): Mask to apply to the grid.
            computeFlexion (bool, optional): If True, compute flexion maps.

        Returns:
            None. The computed maps are stored as object properties.
        """
        if thetax is None or thetay is None:
            self.theta1, self.theta2 = np.meshgrid(theta, theta)
            self.thetax = theta
            self.thetay = theta
        else:
            self.theta1, self.theta2 = np.meshgrid(thetax, thetay)
            if np.round(thetax[1] - thetax[0], 6) == np.round(thetay[1] - thetay[0], 6):
                self.thetax = thetax
                self.thetay = thetay
            else:
                raise Exception('thetax and thetay must have the same pixel scale '
                                + str(thetax[1] - thetax[0])
                                + ', ' + str(thetay[1] - thetay[0]))

        self.mask = mask
        self.perturbed = False

        if self.mask is not None:
            self.a1 = np.full(self.theta1.shape, np.nan)
            self.a2 = np.full(self.theta1.shape, np.nan)
            self.ka = np.full(self.theta1.shape, np.nan)
            self.g1 = np.full(self.theta1.shape, np.nan)
            self.g2 = np.full(self.theta1.shape, np.nan)

            self.theta1 = self.theta1[self.mask]
            self.theta2 = self.theta2[self.mask]

            self.compute_maps_with_mask(computeFlexion)
        else:
            self.compute_maps_without_mask(computeFlexion)

        self.size1 = np.max(self.thetax) - np.min(self.thetax)
        self.size2 = np.max(self.thetay) - np.min(self.thetay)
        self.nray1 = len(self.thetax)
        self.nray2 = len(self.thetay)
        self.pixel_scale = self.thetax[1] - self.thetax[0]

        if self.computed_potential:
            self.conv_fact_time = (
                (1. + self.zl) / c.to(u.km / u.s) *
                (self.dl * self.ds / self.dls).to(u.km)).to(u.d) * \
                (np.pi / 180.0 / 3600.) ** 2

        if not self.computed_potential and compute_potential:
            self.potential_from_kappa()
            self.computed_potential = True
            self.conv_fact_time = (
                (1. + self.zl) / c.to(u.km / u.s) *
                (self.dl * self.ds / self.dls).to(u.km)).to(u.d) * \
                (np.pi / 180.0 / 3600.) ** 2

    def compute_maps_with_mask(self, computeFlexion):
        """
        Compute lens maps (shear, convergence, deflection angles, flexion) for masked grid.

        Args:
            computeFlexion (bool): If True, compute flexion maps.

        Returns:
            None
        """
        self.g1[self.mask], self.g2[self.mask] = self.gamma(self.theta1, self.theta2)
        self.ka[self.mask] = self.kappa(self.theta1, self.theta2)
        self.a1[self.mask], self.a2[self.mask] = self.angle(self.theta1, self.theta2)
        if computeFlexion:
            g11 = np.full(self.theta1.shape, np.nan)
            g22 = np.full(self.theta1.shape, np.nan)
            g12 = np.full(self.theta1.shape, np.nan)
            g21 = np.full(self.theta1.shape, np.nan)
            self.F1 = np.full(self.theta1.shape, np.nan)
            self.F2 = np.full(self.theta1.shape, np.nan)
            self.G1 = np.full(self.theta1.shape, np.nan)
            self.G2 = np.full(self.theta1.shape, np.nan)
            g12[self.mask], g11[self.mask] = np.gradient(self.g1)
            g22[self.mask], g21[self.mask] = np.gradient(self.g2)
            self.F1[self.mask], self.F2[self.mask] = g11 + g22, g21 - g12
            self.G1[self.mask], self.G2[self.mask] = g11 - g22, g21 + g12

    def compute_maps_without_mask(self, computeFlexion):
        """
        Compute lens maps (shear, convergence, deflection angles, flexion) for unmasked grid.

        Args:
            computeFlexion (bool): If True, compute flexion maps.

        Returns:
            None
        """
        self.g1, self.g2 = self.gamma(self.theta1, self.theta2)
        self.ka = self.kappa(self.theta1, self.theta2)
        self.a1, self.a2 = self.angle(self.theta1, self.theta2)
        if computeFlexion:
            g12, g11 = np.gradient(self.g1)
            g22, g21 = np.gradient(self.g2)
            self.F1, self.F2 = g11 + g22, g21 - g12
            self.G1, self.G2 = g11 - g22, g21 + g12

    def tancl(self, size_principale=5.0):
        """
        Find tangential critical lines (sorted by area).

        Args:
            size_principale (float): Minimum size for main critical lines.

        Returns:
            np.ndarray: List of CriticalLine objects, sorted by equivalent Einstein radius.
        """
        lambdat=1.0-self.ka-np.sqrt(self.g1*self.g1+self.g2*self.g2)
        contours = measure.find_contours(lambdat, 0.0)
        lc_all = []
        j=0
        for contour in contours:
            contour[:, 0], contour[:, 1] = contour[:, 1].copy(), contour[:, 0].copy()
            ls = shapely.geometry.LineString(contour)
            lr = shapely.geometry.LineString(ls.coords[:] + ls.coords[0:1])
            mls = unary_union(lr)
            mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
            lc = CriticalLine(j, mp)
            lc.setPoints(contour)
            A = 0.0
            for g in range(len(mp.geoms)):
                A += mp.geoms[g].area
            lc.setArea(A)
            if lc.getThetaE()*self.pixel_scale > size_principale:
                lc.principale = True
            point = shapely.geometry.Point(self.nray1/2., self.nray2/2)
            if (mp.contains(point)):
                lc.principale = True
            lc_all.append(lc)
            j+=1
        cl = sorted(lc_all, key=lambda x: x.getArea(), reverse=True)
        return(np.array(cl))

    def radcl(self, size_principale=1.0):
        """
        Find radial critical lines (sorted by area).

        Args:
            size_principale (float): Minimum size for main critical lines.

        Returns:
            np.ndarray: List of CriticalLine objects, sorted by equivalent Einstein radius.
        """
        lambdar=1.0-self.ka+np.sqrt(self.g1*self.g1+self.g2*self.g2)
        contours = measure.find_contours(lambdar, 0.0)
        lc_all = []
        j=0
        for contour in contours:
            contour[:, 0], contour[:, 1] = contour[:, 1], contour[:, 0].copy()
            ls = shapely.geometry.LineString(contour)
            lr = shapely.geometry.LineString(ls.coords[:] + ls.coords[0:1])
            mls = unary_union(lr)
            mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
            lc = CriticalLine(j, mp)
            lc.setPoints(contour)
            A = 0.0
            for g in range(len(mp.geoms)):
                A += mp.geoms[g].area
            lc.setArea(A)
            if lc.getThetaE()*self.pixel_scale > size_principale:
                lc.principale = True
            point = shapely.geometry.Point(self.nray1/2., self.nray2/2)
            if (mp.contains(point)):
                lc.principale = True
            lc_all.append(lc)
            j+=1
        cl = sorted(lc_all, key=lambda x: x.getArea(), reverse=True)
        return(np.array(cl))

    def getCritPoints(self, lc, pixel_units=False):
        """
        Get coordinates of critical points from a CriticalLine object.

        Args:
            lc (CriticalLine): Critical line object.
            pixel_units (bool): If True, return pixel units.

        Returns:
            tuple: (x1, x2) coordinates of critical points.
        """

        vs=lc.points
        x1,x2=zip(*vs)
        if (pixel_units):
            return(np.array(x1),np.array(x2))
        else:
            #x1=(np.array(x1)-self.a1.shape[1]/2.0)*self.pixel_scale
            #x2=(np.array(x2)-self.a1.shape[0]/2.0)*self.pixel_scale
            x1 = np.array(x1)*self.pixel_scale + self.thetax[0]
            x2 = np.array(x2)*self.pixel_scale + self.thetay[0]
            return(np.array(x1),np.array(x2))

    def mapCrit2Cau(self, x1, x2):
        """
        Map critical points to caustic points.

        Args:
            x1 (np.ndarray): Critical point x-coordinates.
            x2 (np.ndarray): Critical point y-coordinates.

        Returns:
            tuple: (y1, y2) caustic point coordinates.
        """
        a1 = map_coordinates(self.a1, [[x2], [x1]], order=1, prefilter=True)
        a2 = map_coordinates(self.a2, [[x2], [x1]], order=1, prefilter=True)
        y1 = x1 - a1[0]/self.pixel_scale
        y2 = x2 - a2[0]/self.pixel_scale
        return (y1, y2)

    def getCaustics(self, lc_ord, buf_size=0.0):
        """
        Get caustics from a list of CriticalLine objects.

        Args:
            lc_ord (list): List of CriticalLine objects.
            buf_size (float, optional): Buffer size for caustics.

        Returns:
            np.ndarray: List of Caustic objects.
        """
        cau_ord=np.array([])
        for i in range(len(lc_ord)):
            x1,x2=self.getCritPoints(lc_ord[i],pixel_units=True)
            y1,y2=self.mapCrit2Cau(x1,x2)
            if (y1.size>1):
                points=list(zip(y1,y2))
                points=np.asarray(points)
            else:
                points=[(0, 0), (0, 0)]
            ls = shapely.geometry.LineString(points)
            lr = shapely.geometry.LineString(ls.coords[:] + ls.coords[0:1])
            mls = unary_union(lr)
            mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
            cau=Caustic(lc_ord[i].ID,mp)
            cau.principale=lc_ord[i].principale
            cau.passsize=lc_ord[i].passsize
            cau.setPoints(points)
            cau.sigmaloc=lc_ord[i].sigmaloc
            if buf_size > 0.0:
                cau.addBuffer(buffer_size=buf_size)
            A=mp.area
            cau.setArea(A)
            cau_ord=np.append(cau_ord,cau)
        return(cau_ord)

    def getCausticPoints(self, cau, pixel_units=False):
        """
        Get caustic point coordinates from a Caustic object.

        Args:
            cau (Caustic): Caustic object.
            pixel_units (bool): If True, return pixel units.

        Returns:
            tuple: (y1, y2) caustic point coordinates.
        """
        vs=cau.points
        y1,y2=zip(*vs)
        if (pixel_units):
            return(y1,y2)
        else:
            #y1=(np.array(y1)-self.a1.shape[0]/2.0)*self.pixel_scale
            #y2=(np.array(y2)-self.a1.shape[1]/2.0)*self.pixel_scale
            y1 = np.array(y1) * self.pixel_scale + self.thetax[0]
            y2 = np.array(y2) * self.pixel_scale + self.thetay[0]
        return(y1,y2)

    def getBufferCausticPoints(self, cau, pixel_units=False):
        """
        Get buffer caustic point coordinates from a Caustic object.

        Args:
            cau (Caustic): Caustic object.
            pixel_units (bool): If True, return pixel units.

        Returns:
            tuple: (y1, y2) buffer caustic point coordinates.
        """
        if cau.with_buffer:
            y1,y2 = cau.buffer.exterior.coords.xy
            if (pixel_units):
                return(y1,y2)
            else:
                y1 = np.array(y1) * self.pixel_scale + self.thetax[0]
                y2 = np.array(y2) * self.pixel_scale + self.thetay[0]
            return(y1,y2)
        else:
            raise RuntimeError("The caustic does not have any buffer.")

    def combinewith(self, gl):
        """
        Add another lens model to the current lens.

        Args:
            gl (genlen): Another genlen object.

        Returns:
            None. Updates convergence, shear, and deflection angles.
        """
        if np.round(gl.pixel_scale,6) != np.round(self.pixel_scale,6)\
                or gl.size1 != self.size1 \
                or gl.size2 != self.size2:
            raise Exception('Incompatible sizes of deflectors (conbinewith)')

        if gl.mask is not None:
            self.ka[gl.mask]=self.ka[gl.mask]+gl.ka[gl.mask]
            self.g1[gl.mask]=self.g1[gl.mask]+gl.g1[gl.mask]
            self.g2[gl.mask]=self.g2[gl.mask]+gl.g2[gl.mask]
            self.a1[gl.mask]=self.a1[gl.mask]+gl.a1[gl.mask]
            self.a2[gl.mask]=self.a2[gl.mask]+gl.a2[gl.mask]
        else:
            self.ka=self.ka+gl.ka
            self.g1=self.g1+gl.g1
            self.g2=self.g2+gl.g2
            self.a1=self.a1+gl.a1
            self.a2=self.a2+gl.a2

    def add_perturber(self,pb):
        """
        Add a perturber object to the lens model.

        Args:
            pb: Perturber object to be added to the lens model.

        Returns:
            None. Sets the perturber and marks the lens as perturbed.
        """
        self.pb = pb
        self.perturbed = True

#    def kpad(self):
#        """
#        Zero pad the convergence map
#        :return: zero padded convergence map
#        """
#        def padwithzeros(vector, pad_width, iaxis, kwargs):
#            vector[:pad_width[0]] = 0
#            vector[-pad_width[1]:] = 0
#            return vector
#        return np.lib.pad(self.ka, 2*self.ka.shape[0],padwithzeros)
    def kpad(self,kappamap=None):
        """
        Zero pad the convergence map
        :return: zero padded convergence map
        """
        n0 = int(self.ka.shape[0])*2
        n1 = int(self.ka.shape[1])*2
        if kappamap is None:
            return np.pad(self.ka, ((n0,n0),(n1,n1)),
                          mode='constant', constant_values = 0.0)
        else:
            return np.pad(kappamap, ((n0,n0),(n1,n1)), mode='constant',
                          constant_values = 0.0)

    def potential_from_kappa(self):
        """
        Compute the lensing potential from the convergence map
        :return: None (the lensing potential is stored in the attribute
                 pot of the current lens model)
        """
        import warnings
        if self.ka.shape[0] != self.ka.shape[1]:
            warnings.warn("Warning: potential calculation with rectangular grids was not yet tested")
        kappa_padded = self.kpad()
        # define an array of wavenumbers (two components k1,k2)
        k = np.array(np.meshgrid(fftengine.fftfreq(kappa_padded.shape[1])\
                                 ,fftengine.fftfreq(kappa_padded.shape[0])))
        #Compute Laplace operator in Fourier space = -4*pi*k^2
        kk = k[0]**2 + k[1]**2
        kk[kk==0.0] = 1e-10#0.0
        #FFT of the convergence
        kappa_ft = fftengine.fftn(kappa_padded)
        #compute the FT of the potential
        kappa_ft *= - 1.0 / (kk * (2.0*np.pi**2))
        #kappa_ft[0,0] = 0.0
        potential=fftengine.ifftn(kappa_ft) #units should be rad**2
        self.pot=self.mapCrop(potential.real)*self.pixel_scale**2

    def potential_from_kappa1(self,pad_factor=2):
        """
        Compute the lensing potential from the convergence map
        :return: None (the lensing potential is stored in the attribute
                    pot of the current lens model)
        """
        import warnings
        from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
        if self.ka.shape[0] != self.ka.shape[1]:
            warnings.warn("Warning: potential calculation with rectangular grids was not yet tested")

        # Zero padding
        pad_width = 2*int(self.ka.shape[0])
        kappa_padded = self.kpad()
        self.kappa_padded = kappa_padded

        kappa_ft = fft2(kappa_padded)
        kappa_ft_shifted = fftshift(kappa_ft)
        ny, nx = kappa_padded.shape
        kx = np.fft.fftfreq(nx, self.pixel_scale)
        ky = np.fft.fftfreq(ny, self.pixel_scale)
        kx, ky = np.meshgrid(kx, ky)
        kx = fftshift(kx)
        ky = fftshift(ky)

        # Compute the Fourier transform of the potential
        denominator = (2 * np.pi) ** 2 * (kx ** 2 + ky ** 2)
        denominator[denominator == 0.0] = 1e-10  # Avoid division by zero at the zero frequency
        psi_ft = -2 * kappa_ft_shifted / denominator
        #psi_ft[0, 0] = 0  # Set the zero frequency component to zero

        # Inverse FFT to get back to real space
        psi_padded = np.real(ifft2(ifftshift(psi_ft)))
        psi = psi_padded[pad_width:-pad_width, pad_width:-pad_width]
        self.pot = psi#self.mapCrop(psi)


    def mapCrop(self,mappa):
        """
        Crop the map by removing the region added for zero padding
        :param mappa: input map to be cropped (e.g. the lensing potential
        :return: cropped map
        """
        nx = self.ka.shape[0]*2
        ny = self.ka.shape[1]*2
        mappa = mappa[nx:-nx,ny:-ny]
        return(mappa)

    def t_geom_surf(self, beta=None):
        """
        Computes the geometrical time delay
        :param beta: array of size 2 containing the coordinates of the source
        :return: surface of the geometrical time delay
        """
        if (not self.computed_potential):
            raise RuntimeError(
                "Cannot compute the geometrical time delay surface because the potential was not computed. "
                "Enable potential calculation when creating the lens instance (compute_potential=True)."
            )
        x = np.asarray(self.thetax, dtype=float)[None, :]
        y = np.asarray(self.thetay, dtype=float)[:, None]
        if beta is None:
            x0 = 0.0
            y0 = 0.0
        else:
            x0 = beta[0]
            y0 = beta[1]

        return (0.5 * ((x - x0) * (x - x0) + (y - y0) * (y - y0)))

    def t_grav_surf(self):
        """
        Computes the gravitational time delay
        :return: surface of the gravitational time delay
        """
        if (not self.computed_potential):
            raise RuntimeError(
                "Cannot compute the gravitational time delay surface because the potential was not computed. "
                "Enable potential calculation when creating the lens instance (compute_potential=True)."
            )
        return -self.pot

    def t_delay_surf(self,beta=None):
        """
        Computes the total time delay surface, given a source position
        :param beta: coordinates of the source
        :return: numpy.array containing the map of the time delay surface
        """
        if (not self.computed_potential):
            raise RuntimeError(
                "Cannot compute the time delay surface because the potential was not computed. "
                "Enable potential calculation when creating the lens instance (compute_potential=True)."
            )
        t_grav=self.t_grav_surf()
        t_geom=self.t_geom_surf(beta)
        return (t_grav+t_geom)*self.conv_fact_time.value

    def change_redshift(self,newzs):
        """
        Change the source redshift and update all relevant lensing quantities.

        Args:
            newzs (float): New source redshift.

        Returns:
            None. Updates internal properties for the new redshift.
        """
        ds_ = self.co.angular_diameter_distance(newzs)
        dls_ = self.co.angular_diameter_distance_z1z2(self.zl, newzs)
        if self.computed_potential:
            self.pot = self.pot*self.ds.value/self.dls.value*dls_.value/ds_.value
            self.conv_fact_time=self.conv_fact_time/(self.ds.value/self.dls.value*dls_.value/ds_.value)
        self.ka=self.ka*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.g1=self.g1*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.g2=self.g2*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.a1=self.a1*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.a2=self.a2*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.zs=newzs
        self.ds=ds_
        self.dls=dls_

    def listPolygons(self,calines_t,append_itself=False):
        """
        Create a list of all polygons that make up a caustic.

        Args:
            calines_t (list): List of caustic line objects.
            append_itself (bool, optional): If True, append the union of the caustic itself.

        Returns:
            list: List of shapely Polygon objects.
        """
        polygons = []
        for ct in calines_t:
            if (ct.geometria.geom_type == 'MultiPolygon'):
                tmp = list(ct.geometria)
                if append_itself:
                    polygons.append(unary_union(ct.geometria))
                for subpol in tmp:
                    polygons.append(unary_union(subpol))
                    #polygons.append(subpol.buffer(0))
            elif (ct.geometria.geom_type == 'Polygon'):
                polygons.append(unary_union(ct.geometria))
        return polygons

    def multImaCrossSection(self, buffer_size=0.0, plotUU=False):
        """
        Compute the cross section for multiple images, i.e., the area enclosed by the caustics.

        This method allows inclusion of a buffer around the caustics to account for extended sources.
        Optionally, the cross section can be visualized using matplotlib.

        Args:
            buffer_size (float, optional): Buffer size to add around caustics (default is 0.0).
            plotUU (bool, optional): If True, plot the cross section using matplotlib (default is False).

        Returns:
            float: Cross section for multiple images in arcsec^2.
        """

        self.UU = self.causticsUnaryUnion(buffer_size=buffer_size)
        if plotUU:
            ax = self.plot_shapely_polygon()
            ax.set_title('Cross section for multiple images')
            plt.show()

        return self.UU.area * self.pixel_scale ** 2

    def points_in_UU(self,x_coords, y_coords):
        """
        Check which points are inside the unary union UU.

        Args:
            x_coords (array): X coordinates.
            y_coords (array): Y coordinates.

        Returns:
            np.ndarray: Boolean array with True for points inside UU.
        """
        mask = []
        for x, y in zip(x_coords, y_coords):
            point = Point(x, y)
            mask.append(self.UU.contains(point))
        return np.array(mask)

    def plot_shapely_polygon(self, ax=None, color='blue', alpha=0.4):
        """
        Plot the shapely polygon (caustic union) on a matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.
            color (str, optional): Polygon color.
            alpha (float, optional): Polygon transparency.

        Returns:
            matplotlib.axes.Axes: The axis with the plotted polygon.
        """
        print("Plotting shapely polygon", self.UU.geom_type)
        if ax is None:
            fig, ax = plt.subplots()
        bounds = []
        def plot_poly(poly):
            if poly.is_empty or poly.area == 0:
                return
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha)
            for interior in poly.interiors:
                x, y = interior.xy
                ax.plot(x, y, color='k')
            bounds.append(poly.bounds)
        if self.UU.geom_type == 'Polygon':
            plot_poly(self.UU)
        elif self.UU.geom_type == 'MultiPolygon':
            for poly in self.UU.geoms:
                plot_poly(poly)
        elif self.UU.geom_type == 'GeometryCollection':
            for geom in self.UU.geoms:
                if geom.geom_type == 'Polygon':
                    plot_poly(geom)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        plot_poly(poly)
        ax.set_aspect('equal')
        if bounds:
            minx = min(b[0] for b in bounds)
            miny = min(b[1] for b in bounds)
            maxx = max(b[2] for b in bounds)
            maxy = max(b[3] for b in bounds)
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
        else:
            print("No non-empty polygons to plot.")
        return ax

    def imageMultiplicity(self):
        """
        Compute regions on the source plane that produce three, five, or seven images.

        This method calculates the caustics and determines the areas where multiple caustic regions overlap,
        corresponding to different image multiplicities. Note: Resonant secondary caustics are not separated from
        primary caustics, so some regions may not be correctly treated for image multiplicity.

        Returns:
            tuple:
                - list of shapely geometries (regions for 3, 5, 7 images)
                - numpy.ndarray: cross sections for each multiplicity in arcsec^2
        """
        # step 1: calculate caustics
        clt = self.tancl()
        clr = self.radcl()
        caut = self.getCaustics(clt)
        caur = self.getCaustics(clr)
        # step 2: concatenate radial and tangential caustics and create a list of polygons
        cau=np.concatenate((caut,caur),axis=None)
        all_cau_geom = [unary_union(cau[i].geometria) for i in range(len(cau))]
        all_cau_geom=sorted(all_cau_geom, key=lambda x: x.area)

        # step 3: find regions where two caustics overlap -> multiplycity 5
        these_intersections=all_cau_geom.copy()
        inter = [unary_union(all_cau_geom)]
        intersections = []
        for i in range(len(these_intersections)):
            this_cau=these_intersections[i]
            for j in range(i+1,len(these_intersections)):
                that_cau=these_intersections[j]
                intersections.append(unary_union(this_cau.intersection(that_cau)))
        inter.append(unary_union(intersections))

        # step 4: find regions where three caustics overlap -> multiplycity 7
        these_intersections=all_cau_geom.copy()
        intersections = []
        for i in range(len(these_intersections)):
            this_cau=these_intersections[i]
            for j in range(i+1,len(these_intersections)):
                that_cau=these_intersections[j]
                for k in range(j+1,len(these_intersections)):
                    this_other_cau = these_intersections[k]
                    intersections.append(unary_union(unary_union(this_cau.intersection(that_cau)).intersection(this_other_cau)))
        inter.append(unary_union(intersections))

        # step 5: calculate cross sections for 3, 5, 7 images
        cross_sections = np.array([inter[0].area-inter[1].area-inter[2].area,
                          inter[1].area-inter[2].area,inter[2].area])*self.pixel_scale**2
        return inter, cross_sections

    def ggslCrossSection(self,clt=None,minsize=0.5,maxsize=5.0,dmax=200.0,buffer_size = 0.0):
        """
        Compute the GGSL (Galaxy-Galaxy Strong Lensing) cross section for the current source redshift.

        This method follows the definition in Meneghetti et al. (2020) and uses the ray grid units.
        It allows filtering critical lines by size, distance, and buffer for extended sources.

        Args:
            clt (list, optional): Array of previously computed CriticalLine objects. If None, they are computed automatically.
            minsize (float, optional): Minimum size of critical lines to include (units of theta).
            maxsize (float, optional): Maximum size of critical lines to include (units of theta).
            dmax (float, optional): Maximum distance from center for critical lines (units of theta).
            buffer_size (float, optional): Buffer size around caustics for extended sources.

        Returns:
            float: GGSL cross section for the current source redshift (units of theta^2).
        """
        if type(clt) == type(None):
            clt = self.tancl()
        dist = np.array([np.sqrt((clt[i].px-self.nray1/2.0)**2
                       +(clt[i].py-self.nray2/2.0)**2)*self.pixel_scale for i in range(len(clt))])
        caut = self.getCaustics(clt)

        area = 0.0
        for i in range(len(clt)):
            if (dist[i] < dmax) & \
                    (clt[i].getThetaE()*self.pixel_scale > minsize) & \
                    (clt[i].getThetaE()*self.pixel_scale < maxsize) & \
                    (clt[i].area<clt[0].area/4.) & \
                    (not clt[i].principale):
                if buffer_size > 0.0:
                    caut[i].addBuffer(buffer_size=buffer_size)
                    area += caut[i].getBufferArea()
                else:
                    area+= caut[i].getArea()
        return area*self.pixel_scale**2

    def thetaE(self,clt=None):
        """
        Compute the Einstein radius for the lens model.

        If no critical lines are provided, they are computed automatically.

        Args:
            clt (list, optional): Previously computed critical lines. If None, computes them automatically.

        Returns:
            float: Einstein radius (units of theta).
        """
        if type(clt) == type(None):
            clt = self.tancl()
        return clt[0].getThetaE()*self.pixel_scale

    def thetaEv(self,clt=None):
        """
        Compute Einstein radii for each critical line detected in the field of view.

        If no critical lines are provided, they are computed automatically. Units are those of theta used to set up the ray grid.

        Args:
            clt (list, optional): Previously computed critical lines. If None, computes them automatically.

        Returns:
            np.ndarray: Array of Einstein radii (units of theta).
        """
        if type(clt) == type(None):
            clt = self.tancl()
        return np.array([clt[i].getThetaE()*self.pixel_scale for i in range(len(clt))])

    def causticRegion(self,caut=None):
        """
        Compute the convex hull of caustic representative points.

        Args:
            caut (list, optional): List of caustic objects. If None, caustics are computed.

        Returns:
            tuple: (ConvexHull object, points array, hull volume) or (None, None, None) if insufficient points.
        """
        if type(caut) == type(None):
            clt = self.tancl()
            caut = self.getCaustics(clt)
        x = []
        y = []
        for c in caut:
            try:
                x.append(c.geometria.representative_point().x)
                y.append(c.geometria.representative_point().y)
            except Exception:
                x.append(0.0)
                y.append(0.0)

        if (len(x)>2):
            x = np.array(x)
            y = np.array(y)
            x = (x - self.nray1 / 2.0) * self.pixel_scale
            y = (y - self.nray2 / 2.0) * self.pixel_scale
            points = np.column_stack((x, y))
            hull = ConvexHull(points)
            return hull,points,hull.volume
        else:
            return None,None,None

    def fovSP(self):
        """
        Compute the border of the field of view in the source plane and its area.

        Returns:
            tuple: (border x coordinates, border y coordinates, sampled area)
        """
        border = np.arange(0, self.nray1 - 1, 1)
        zeros = border * 0.0
        npixers = border * 0.0 + self.nray1 - 1
        border_x = np.concatenate([border, npixers, np.flipud(border), zeros])

        border = np.arange(0, self.nray2 - 1, 1)
        zeros = border * 0.0
        npixers = border * 0.0 + self.nray2 - 1
        border_y = np.concatenate([zeros, border, npixers, np.flipud(border)])

        sp_borderx, sp_bordery = self.mapCrit2Cau(border_x, border_y)
        vs = zip(sp_borderx, sp_bordery)
        ls = shapely.geometry.LineString(vs)
        lr = shapely.geometry.LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(ls)
        mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
        sampled_area = mp.area * self.pixel_scale ** 2
        return (sp_borderx-self.nray1/2.)*self.pixel_scale, (sp_bordery-self.nray2/2.)*self.pixel_scale, sampled_area

    def fovSP_from_x1x2(self, x1, x2):
        """
        Compute the border of the field of view in the source plane and its area from given x1, x2 coordinates.

        Args:
            x1 (np.ndarray): X coordinates in the lens plane.
            x2 (np.ndarray): Y coordinates in the lens plane.
        Returns:
            tuple: (border x coordinates, border y coordinates, sampled area)
        """
        # convert x1, x2 to pixel indices
        x1_pix = (x1 / self.pixel_scale) + (self.nray1 / 2.0)
        x2_pix = (x2 / self.pixel_scale) + (self.nray2 / 2.0)

        sp_borderx, sp_bordery = self.mapCrit2Cau(x1_pix, x2_pix)
        vs = zip(sp_borderx, sp_bordery)
        ls = shapely.geometry.LineString(vs)
        lr = shapely.geometry.LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(ls)
        mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
        sampled_area = mp.area * self.pixel_scale ** 2
        return (sp_borderx - self.nray1 / 2.) * self.pixel_scale, (sp_bordery - self.nray2 / 2.) * self.pixel_scale, sampled_area

    def multImaRegion_old(self):
        """
        Returns the region in the lens plane corrisponding to image multiplicity larger than 1.
        THIS FUNCTION IS SUPERSEEDED BY THE NEW **multImaRegion** FUNCTION.
        :return: theta1, theta2: numpy arrays containing the coordinates of the pixels
                                 in the image plane belonging to the region of image multiplicity > 1
        """
        # create array of grid points
        y1 = (self.theta1 - self.a1)  # y1 coordinates on the source plane
        y2 = (self.theta2 - self.a2)
        y1=y1.flatten()
        y2=y2.flatten()

        points = np.vstack((y1, y2)).T
        clt = self.tancl()
        clr = self.radcl()
        caut = self.getCaustics(clt)
        caur = self.getCaustics(clr)
        cau=np.concatenate((caut,caur),axis=None)
        cau = sorted(cau, key=lambda x: x.area)
        multi = np.zeros(len(y1))
        for i in range(len(cau)):
            #print("cau %d of %d" % (i, len(cau)))
            for c in list(cau[i].geometria):
                if c.geom_type == 'Polygon':
                    x, y = c.exterior.xy
                    x = (np.array(x) - self.nray1 / 2.0) * self.pixel_scale
                    y = (np.array(y) - self.nray2 / 2.0) * self.pixel_scale
                    vpoints=np.vstack((x,y)).T
                    p=Path(vpoints)
                    inside = p.contains_points(points,radius=-0.3)
                    multi[inside] += 1


            #p = Path(cau[i].points)
            #inside = p.contains_points(points)
            #multi[inside] += 1

        isel = multi > 0
        theta1_flat = self.theta1.flatten()
        theta2_flat = self.theta2.flatten()

        return theta1_flat[isel], theta2_flat[isel]

    def getAngle(self, theta1, theta2, order=1):
        """
        Compute the deflection angle at a given position by interpolating the deflection angle maps.

        Args:
            theta1 (float): X coordinate (in angular units, e.g., arcsec) where the deflection angle is evaluated.
            theta2 (float): Y coordinate (in angular units, e.g., arcsec) where the deflection angle is evaluated.
            order (int, optional): Interpolation order for map_coordinates (default is 1).

        Returns:
            tuple: (a1, a2) interpolated deflection angles at the specified position.
        """
        # from angle to pixel
        x = (theta1 - self.thetax[0]) / self.pixel_scale
        y = (theta2 - self.thetay[0]) / self.pixel_scale

        a1 = map_coordinates(self.a1, [[y], [x]], order=order, prefilter=True)
        a2 = map_coordinates(self.a2, [[y], [x]], order=order, prefilter=True)
        return a1, a2

    def getAngleFromPotential(self, theta1, theta2, order=1):
        """
        Compute the deflection angle implied by the stored lensing potential.

        This is useful when positions must be consistent with the plotted
        time-delay surface, whose stationary points are determined by the
        gradient of the same potential.
        """
        if not self.computed_potential:
            raise RuntimeError(
                "Cannot compute potential-derived deflection because the potential was not computed or loaded."
            )

        grad_y, grad_x = np.gradient(self.pot, self.pixel_scale, self.pixel_scale)
        x = (theta1 - self.thetax[0]) / self.pixel_scale
        y = (theta2 - self.thetay[0]) / self.pixel_scale
        a1 = map_coordinates(grad_x, [[y], [x]], order=order, prefilter=True)
        a2 = map_coordinates(grad_y, [[y], [x]], order=order, prefilter=True)
        return a1, a2

    def approximatedIP(self,betas,dbetas=0.03):
        """
        Find approximate image positions for a given source position using a search radius.

        Args:
            betas (tuple): Source coordinates (beta1, beta2) in angular units (e.g., arcsec).
            dbetas (float, optional): Search radius around the source position (default is 0.03).

        Returns:
            np.ndarray: Array of approximate image positions, each as (theta1, theta2).
        """

        import skimage.morphology
        beta1 = self.theta1 - self.a1
        beta2 = self.theta2 - self.a2

        isel = (beta1-betas[0])**2 + (beta2-betas[1])**2 <= dbetas**2

        labeled = skimage.morphology.label(isel)#(dense)
        coords_theta1theta2 = [(labeled == i).nonzero() for i in range(1, labeled.max() + 1)]
        selthetai = np.array(coords_theta1theta2)*self.pixel_scale

        for i in range(selthetai.shape[0]):
            selthetai[i][0]=selthetai[i][0]-self.nray1*self.pixel_scale/2.0
            selthetai[i][1]=selthetai[i][1]-self.nray2*self.pixel_scale/2.0

        return selthetai

    def random_sources_in_caustic(self,ca,buffer_size=0.0,number=10000):
        """
        Randomly place sources inside a given caustic, optionally with a buffer.

        Args:
            ca: Caustic instance.
            buffer_size (float, optional): Buffer size around caustic in pixels.
            number (int, optional): Number of random sources to generate.

        Returns:
            tuple: Arrays of x and y coordinates of drawn sources.
        """

        pp = ca.random_points_in_caustic(buffer_size=buffer_size, number=number)
        px = pp.points.x.values * self.pixel_scale + self.thetax[0]
        py = pp.points.y.values * self.pixel_scale + self.thetay[0]
        return px, py

    def random_sources_along_caustic(self,ca,buffer_size=1.0,number=10000):
        """
        Randomly place sources in a buffer along a given caustic.

        Args:
            ca: Caustic instance.
            buffer_size (float, optional): Buffer size around caustic in pixels.
            number (int, optional): Number of random sources to generate.

        Returns:
            tuple: Arrays of x and y coordinates of drawn sources.
        """

        pp = ca.random_points_along_caustic(buffer_size=buffer_size, number=number)
        px = pp.points.x.values * self.pixel_scale + self.thetax[0]
        py = pp.points.y.values * self.pixel_scale + self.thetay[0]
        return px, py

    def causticsUnaryUnion(self, buffer_size=0.0):
        """
        Create a shapely unary union of lens caustics, optionally with a buffer.

        Args:
            buffer_size (float, optional): Buffer size in pixels.

        Returns:
            shapely.geometry: Unary union of caustics (possibly buffered).
        """
        clt = self.tancl()
        clr = self.radcl()
        caut = self.getCaustics(clt)
        caur = self.getCaustics(clr)
        cau = np.concatenate((caut, caur), axis=None)

        polygons = [cau[i].geometria for i in range(len(cau))]
        UU = unary_union(polygons)
        if np.abs(buffer_size) > 0.0:
            concaustics = UU.buffer(buffer_size)  # Polygon(UU.buffer(buffer_size))
        else:
            concaustics = UU
        return concaustics

    def multImaRegion(self,buffer_size=0.0):
        """
        Returns the region in the lens plane corresponding to image multiplicity larger than 1, with optional buffer.

        Args:
            buffer_size (float, optional): Buffer size in pixels.

        Returns:
            tuple: Arrays of x and y coordinates of pixels in the region.
        """

        # create array of grid points
        y1 = (self.theta1 - self.a1)  # y1 coordinates on the source plane
        y2 = (self.theta2 - self.a2)
        y1=y1.flatten()
        y2=y2.flatten()
        x1=self.theta1.flatten()
        x2=self.theta2.flatten()

        concaustics = self.causticsUnaryUnion(buffer_size=buffer_size)

        inds_ = self.select_points_in_caustics(y1,y2,concaustics,buffer_size=buffer_size)
        inds=np.array(inds_)

        return x1[inds], x2[inds]

    def select_points_in_caustics(self,px,py,cc,buffer_size=0.0):
        """
        Select points inside the caustic union (with optional buffer).

        Args:
            px (array): X coordinates.
            py (array): Y coordinates.
            cc: Caustic union geometry.
            buffer_size (float, optional): Buffer size.

        Returns:
            np.ndarray: Boolean array indicating which points are inside the caustic region.
        """
        if gpd is None:
            raise ImportError("geopandas is required for select_points_in_caustics().")

        x = (px - self.thetax[0]) / self.pixel_scale
        y = (py - self.thetay[0]) / self.pixel_scale

        buffer = cc.buffer(buffer_size)

        gdf_poly = gpd.GeoDataFrame(index=["myPoly"], geometry=[buffer])
        df = pd.DataFrame()
        df['points'] = list(zip(x, y))
        df['points'] = df['points'].apply(Point)
        gdf_points = gpd.GeoDataFrame(df, geometry='points')
        Sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate="within", how='left')
        inds = Sjoin.index_right == 'myPoly'

        return inds.values
