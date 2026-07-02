import numpy as np
import astropy.io.fits as fits
from astropy.constants import c, G
import scipy.fftpack as fftengine
from scipy.ndimage import map_coordinates
from astropy import units as u
from pyLensLib.convergence_integrals import (
    deflection_from_kappa_grid,
    deflection_from_kappa_grid_adaptive,
    potential_from_kappa_grid,
    potential_from_kappa_grid_adaptive,
)

class raytracer(object):
    """
    Class for tracing rays through a mass map for gravitational lensing simulations.

    Attributes:
        zs (float): Source redshift.
        zl (float): Lens redshift.
        mass (ndarray): Mass map of the lens.
        fov (float): Field of view of the mass map.
        nray (int): Number of rays per axis.
        fov_ray (float): Field of view for ray grid.
        pixel_ray (float): Pixel scale for ray grid.
        co: Cosmology instance.
        dl, ds, dls: Angular diameter distances.
        kappa (ndarray): Convergence map.
        pixel_scale (float): Pixel scale of mass map.
        pot (ndarray): Lensing potential map.
        a1, a2 (ndarray): Deflection angle components.
    """
    def __init__(self,co,massmap,Nray,FOVray,fromfile=True,**kwargs):
        """
        Initialize a raytracer instance for lensing simulation.

        Args:
            co: Cosmology instance.
            massmap (ndarray or str): Mass map array or FITS filename.
            Nray (int): Number of rays per axis.
            FOVray (float): Field of view for ray grid.
            fromfile (bool): If True, read mass map from FITS file.
            **kwargs: Additional parameters (fov, zl, zs).

        Returns:
            None
        """
        self.zs = kwargs.get('zs', 2.0)
        self.method = kwargs.get('method', 'potential_fft')
        self.alpha_interpolation_order = kwargs.get('alpha_interpolation_order', 3)
        self.adaptive_low_res_factor = kwargs.get('low_res_factor', 4)
        self.adaptive_high_res_kernel_size = kwargs.get('high_res_kernel_size', 21)
        self.adaptive_return_high_res = kwargs.get('adaptive_return_high_res', True)
        if fromfile:
            try:
                with fits.open(massmap) as hdul:
                    self.mass = np.float64(hdul[0].data.copy())
                    if self.mass.shape[0] == self.mass.shape[1]:
                        #self.fov = hdul[0].header['CDELT2']*3600.0*(self.mass.shape[0])
                        self.fov = kwargs['fov']
                        self.zl = hdul[0].header['ZL']
                    else:
                        raise ValueError('massmap needs to be a squared array')
            except Exception as exc:
                raise ValueError('Missing/invalid information in FITS file (data, fov, ZL).') from exc
        else:
            try:
                self.mass = massmap
                if self.mass.shape[0] == self.mass.shape[1]:
                    self.fov = kwargs['fov']
                    self.zl = kwargs['zl']
                else:
                    raise ValueError('massmap needs to be a squared array')
            except Exception as exc:
                raise ValueError('Missing information in kwargs or massmap is not a valid array.') from exc
        # ray grid parameters: nray, fov_ray, pixel_ray
        self.nray = Nray # was was self.npix
        self.fov_ray = FOVray # was self.size
        self.pixel_ray = float(self.fov_ray) / float(self.nray-1) # was self.pixel
        self.co = co
        self.dl = self.co.angular_diameter_distance(self.zl)
        self.ds = self.co.angular_diameter_distance(self.zs)
        self.dls = self.co.angular_diameter_distance_z1z2(self.zl, self.zs)

        # mass grid parameters: npix, fov, pixel_scale
        fov_mpc = self.fov*self.dl.value*np.pi/3600.0/180.0
        self.npix = self.mass.shape[0] # was nx, ny
        dx = fov_mpc / (self.npix - 1)
        self.kappa_map = self.mass / dx ** 2 /self.sigma_cr()
        self.kappa = self.kappa_map.copy()
        self.pixel_scale = self.fov / (self.npix-1) # size of the pixel

        self.compute_lensing_maps()

    def compute_lensing_maps(self):
        """
        Compute ray-grid lensing maps with the selected numerical backend.
        """
        method = self.method.lower()
        if method in ('potential_fft', 'legacy', 'fft_potential'):
            # zero-pad the convergence map, preserving the historical backend
            self.kpad()
            self.potential()
        elif method in ('deflection_fft', 'fft_deflection', 'alpha_fft'):
            self.deflection_from_kappa_fft(adaptive=False)
        elif method in ('deflection_fft_adaptive', 'adaptive_fft', 'alpha_fft_adaptive'):
            self.deflection_from_kappa_fft(adaptive=True)
        else:
            raise ValueError(
                "Unknown raytracer method '{}'. Use 'potential_fft', "
                "'deflection_fft', or 'deflection_fft_adaptive'.".format(self.method)
            )

    def kpad(self):
        """
        Zero pad the convergence map.

        Returns:
            None
        """
        pad_width = 2 * self.kappa.shape[0]
        self.kappa = np.pad(
            self.kappa,
            pad_width,
            mode='constant',
            constant_values=0.0,
        )

    def potential_from_kappa(self):
        """
        Solve the 2D Poisson equation to transform the convergence into a lensing potential map.

        Returns:
            ndarray: Map of the lensing potential covering the same FOV as the input convergence map.
        """
        # define an array of wavenumbers (two components k1,k2)
        k = np.array(np.meshgrid(fftengine.fftfreq(self.kappa.shape[0])\
                                 ,fftengine.fftfreq(self.kappa.shape[1])))
        #Compute Laplace operator in Fourier space = -4*pi*k^2
        kk = k[0]**2 + k[1]**2
        kk[0,0] = 1e-8#0.0
        #FFT of the convergence
        kappa_ft = fftengine.fftn(self.kappa)
        #compute the FT of the potential
        kappa_ft *= - 1.0 / (kk * (2.0*np.pi**2))
        #kappa_ft[0,0] = 0.0
        potential=fftengine.ifftn(kappa_ft)
        pot=self.mapCrop(potential.real)
        return pot

    def potential(self):
        """
        Compute and interpolate the lensing potential map over the ray grid.

        Returns:
            None
        """
        x_ = np.linspace(0, self.nray - 1, self.nray)
        y_ = np.linspace(0, self.nray - 1, self.nray)
        x, y = np.meshgrid(x_, y_)
        potential = self.potential_from_kappa()
        # zero coordinate of the ray grid onto the potential map:
        x0 = y0 = (potential.shape[0]) / 2.0 * self.pixel_scale - self.fov_ray / 2.0
        x = (x0 + x  * self.pixel_ray) / self.pixel_scale # coordinate in pixel units on the potential map
        y = (y0 + y  * self.pixel_ray) / self.pixel_scale
        # trick to re-center the ray map in such a way that the centers of the ray and of the mass
        # maps are coincident
        # physical center in the new map:
        xc_new = (x.max() + x.min()) * 0.5 * self.pixel_scale
        yc_new = (y.max() + y.min()) * 0.5 * self.pixel_scale
        # shift with respect to the pysical center of the mass map
        deltax = (xc_new - self.fov/2.0)/self.pixel_scale
        deltay = (yc_new - self.fov/2.0)/self.pixel_scale
        #
        x = x - deltax
        y = y - deltay

        #interpolate the potential map at the ray positions
        pot = map_coordinates(potential, [y, x], order=3,  prefilter=True)#.reshape(int(self.nray), int(self.nray))
        self.pot = pot * self.pixel_scale ** 2 / self.pixel_ray / self.pixel_ray
        self.a2, self.a1 = np.gradient(self.pot)
        self.a1 = self.a1 * self.pixel_ray
        self.a2 = self.a2 * self.pixel_ray
        self.pot = pot * self.pixel_scale ** 2

    def _ray_grid_pixel_coordinates(self, map_shape, map_pixel_scale):
        """
        Coordinates of the requested ray grid in the pixel frame of a map.
        """
        x_ = np.linspace(0, self.nray - 1, self.nray)
        y_ = np.linspace(0, self.nray - 1, self.nray)
        x, y = np.meshgrid(x_, y_)

        x0 = y0 = map_shape[0] / 2.0 * map_pixel_scale - self.fov_ray / 2.0
        x = (x0 + x * self.pixel_ray) / map_pixel_scale
        y = (y0 + y * self.pixel_ray) / map_pixel_scale

        xc_new = (x.max() + x.min()) * 0.5 * map_pixel_scale
        yc_new = (y.max() + y.min()) * 0.5 * map_pixel_scale
        deltax = (xc_new - self.fov / 2.0) / map_pixel_scale
        deltay = (yc_new - self.fov / 2.0) / map_pixel_scale
        return x - deltax, y - deltay

    def _interpolate_to_ray_grid(self, mappa, map_pixel_scale, order=3):
        """
        Interpolate a square map onto the requested ray grid.
        """
        x, y = self._ray_grid_pixel_coordinates(mappa.shape, map_pixel_scale)
        return map_coordinates(mappa, [y, x], order=order, prefilter=True)

    def deflection_from_kappa_fft(self, adaptive=False):
        """
        Compute deflection maps directly from convergence by FFT convolution.
        """
        if adaptive:
            a1_map, a2_map = deflection_from_kappa_grid_adaptive(
                self.kappa_map,
                self.pixel_scale,
                low_res_factor=self.adaptive_low_res_factor,
                high_res_kernel_size=self.adaptive_high_res_kernel_size,
                return_high_res=self.adaptive_return_high_res,
            )
            pot_map = potential_from_kappa_grid_adaptive(
                self.kappa_map,
                self.pixel_scale,
                low_res_factor=self.adaptive_low_res_factor,
                high_res_kernel_size=self.adaptive_high_res_kernel_size,
                return_high_res=self.adaptive_return_high_res,
            )
            map_pixel_scale = self.pixel_scale
            if not self.adaptive_return_high_res:
                map_pixel_scale = self.pixel_scale * self.adaptive_low_res_factor
        else:
            a1_map, a2_map = deflection_from_kappa_grid(
                self.kappa_map, self.pixel_scale
            )
            pot_map = potential_from_kappa_grid(self.kappa_map, self.pixel_scale)
            map_pixel_scale = self.pixel_scale

        order = self.alpha_interpolation_order
        self.a1 = self._interpolate_to_ray_grid(a1_map, map_pixel_scale, order=order)
        self.a2 = self._interpolate_to_ray_grid(a2_map, map_pixel_scale, order=order)
        self.pot = self._interpolate_to_ray_grid(pot_map, map_pixel_scale, order=order)

    def sigma_cr(self):
        """
        Calculate the critical surface density for lensing.

        Returns:
            float: Critical surface density for the lens, given its redshift and the source redshift.
        """
        c2_G_Msun_Mpc = (c ** 2 / G).to(u.Msun / u.Mpc)
        sigma_cr = c2_G_Msun_Mpc / (4 * np.pi) * (self.ds / self.dl / self.dls)
        return (sigma_cr.value)

    # crop the maps to remove zero-padded areas and get back to the original
    # region.
    def mapCrop(self,mappa):
        """
        Crop the maps to remove zero-padded areas and return the original region.

        Args:
            mappa (ndarray): Map to be cropped.

        Returns:
            ndarray: Cropped map.
        """
        xmin=int(self.kappa.shape[0]/2-self.npix/2)
        ymin=int(self.kappa.shape[1]/2-self.npix/2)
        xmax=int(xmin+self.npix)
        ymax=int(ymin+self.npix)
        mappa=mappa[xmin:xmax,ymin:ymax]
        return(mappa)
