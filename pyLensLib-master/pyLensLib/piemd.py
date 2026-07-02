from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from astropy import cosmology
import numpy as np
from pyLensLib.sie import sie
import astropy.units as units
from pyLensLib.genlen import genlen
#import uncertainties

import lmfit

def radial_profile(map, nf=10, logscale=False):
    """
    Compute the radial profile of a 2D map.

    Args:
        map (ndarray): 2D array to profile.
        nf (int): Number of radial bins.
        logscale (bool): Use logarithmic binning if True.

    Returns:
        tuple: (profile values, mean radius of bins)
    """
    y, x = np.indices((map.shape))  # first determine radii of all pixels
    center = [map.shape[0] / 2., map.shape[1] / 2.]
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(float)
    # radius of the image.
    r_max = np.max(r) / np.sqrt(2)
    print(r_max)
    if (logscale):
        bins = np.logspace(0, np.log10(r_max), nf)
    else:
        bins = np.linspace(1, r_max, nf)
    ring_value, radius = np.histogram(r, weights=map, bins=bins)
    area_ring = np.pi * (radius[1:] ** 2 - radius[:-1] ** 2)
    radius_mean = 0.5 * (radius[1:] + radius[:-1])
    return (ring_value / area_ring, radius_mean)

class piemd(genlen):

    """
    Class representing a PIEMD (Pseudo-Isothermal Elliptical Mass Distribution) lens model.

    Attributes:
        zl (float): Lens redshift.
        zs (float): Source redshift.
        co: Cosmology instance.
        dl, ds, dls: Angular diameter distances.
        theta_c (float): Core radius.
        pa (float): Position angle.
        q (float): Axis ratio.
        sigma0 (float): Central velocity dispersion.
        x1, x2 (float): Lens center coordinates.
        theta_t (float): Truncation radius.
        s1, s2: SIE models for core and truncation.
        prof_params (list): Profile parameters.
    """
    def __init__(self, co, **kwargs):
        """
        Initialize a PIEMD lens model.

        Args:
            co: Cosmology instance.
            **kwargs: Lens parameters (zl, zs, theta_c, pa, q, sigma0, x1, x2, theta_t).

        Returns:
            None
        """
        super().__init__()
        self.computed_potential = False
        if ('zl' in kwargs):
            self.zl = kwargs['zl']
        else:
            self.zl = 0.3

        if ('zs' in kwargs):
            self.zs = kwargs['zs']
        else:
            self.zs = 1.5

        self.co=co
        self.dl = co.angular_diameter_distance(self.zl)
        self.ds = co.angular_diameter_distance(self.zs)
        self.dls = co.angular_diameter_distance_z1z2(self.zl, self.zs)

        if ('theta_c' in kwargs):
            self.theta_c = kwargs['theta_c']
        else:
            self.theta_c = 0.0

        if ('pa' in kwargs):
            self.pa = kwargs['pa']
        else:
            self.pa = 0.0

        if ('q' in kwargs):
            self.q = kwargs['q']
        else:
            self.q = 1.0

        if ('sigma0' in kwargs):
            self.sigma0 = kwargs['sigma0']
        else:
            self.sigma0 = 200.0

        if ('x1' in kwargs):
            self.x1 = kwargs['x1']
        else:
            self.x1 = 0.0

        if ('x2' in kwargs):
            self.x2 = kwargs['x2']
        else:
            self.x2 = 0.0

        if ('theta_t' in kwargs):
            self.theta_t = kwargs['theta_t']
        else:
            self.theta_t = self.rt()
            self.theta_t = self.theta_t* 1e-3 / self.dl.value * \
                           180.0 / np.pi * 3600.0



        kwargs1 = {'zl': self.zl,
                   'zs': self.zs,
                   'sigma0': self.sigma0,
                   'q': self.q,
                   'theta_c': self.theta_c,
                   'pa': self.pa,
                   'x1': self.x1,
                   'x2': self.x2}
        kwargs2 = {'zl': self.zl,
                   'zs': self.zs,
                   'sigma0': self.sigma0,
                   'q': self.q,
                   'theta_c': self.theta_t,
                   'pa': self.pa,
                   'x1': self.x1,
                   'x2': self.x2}
        self.s1 = sie(co, **kwargs1)
        self.s2 = sie(co, **kwargs2)

        self.prof_params = ['theta_t', 'theta_c', 'sigma0']

    def kappa(self, theta1, theta2):
        """
        Compute the convergence (kappa) at a given position.

        Args:
            theta1 (float or array): x coordinate.
            theta2 (float or array): y coordinate.

        Returns:
            float or array: Convergence value.
        """
        #isel = (theta1 - self.x1)**2+(theta2 - self.x2)**2 < 5.0*self.theta_t
        kappa_ = self.s1.kappa(theta1, theta2) - self.s2.kappa(theta1, theta2)
        return kappa_

    def potential(self, theta1, theta2):
        """
        Compute the lensing potential at a given position.

        Args:
            theta1 (float or array): x coordinate.
            theta2 (float or array): y coordinate.

        Returns:
            float or array: Lensing potential value.
        """
        pot = self.s1.potential(theta1, theta2) - self.s2.potential(theta1, theta2)
        return pot

    def angle(self, theta1, theta2):
        """
        Compute the deflection angle at a given position.

        Args:
            theta1 (float or array): x coordinate.
            theta2 (float or array): y coordinate.

        Returns:
            tuple: Deflection angle components (alphax, alphay).
        """
        alphax_1, alphay_1 = self.s1.angle(theta1, theta2)
        alphax_2, alphay_2 = self.s2.angle(theta1, theta2)
        return (alphax_1 - alphax_2, alphay_1 - alphay_2)

    def gamma(self, theta1, theta2):
        """
        Compute the shear (gamma) at a given position.

        Args:
            theta1 (float or array): x coordinate.
            theta2 (float or array): y coordinate.

        Returns:
            tuple: Shear components (gammax, gammay).
        """
        gammax_1, gammay_1 = self.s1.gamma(theta1, theta2)
        gammax_2, gammay_2 = self.s2.gamma(theta1, theta2)
        gammax = gammax_1 - gammax_2
        gammay = gammay_1 - gammay_2
        return (gammax, gammay)

    def mass(self):
        """
        Compute the total mass of the PIEMD lens.

        Returns:
            float: Total mass.
        """
        GG=const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        return(np.pi*self.sigma0**2/GG *self.theta_t*self.dl.value*np.pi/180.0/3600)

    def m2Dr(self,r):
        """
        Compute the projected mass within radius r.

        Args:
            r (float): Radius.

        Returns:
            float: Projected mass within r.
        """
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        rcut=self.theta_t * self.dl.value * np.pi / 180.0 / 3600
        rcore=self.theta_c * self.dl.value * np.pi / 180.0 / 3600
        return (np.pi * self.sigma0 ** 2 / GG * rcut / (rcut-rcore) *
                (np.sqrt(r**2 + rcore**2) - rcore -
                 np.sqrt(r**2 + rcut**2) + rcut))

    def m3Dr(self,r):
        """
        Compute the 3D mass within radius r.

        Args:
            r (float): Radius.

        Returns:
            float: 3D mass within r.
        """
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        rcut=self.theta_t * self.dl.value * np.pi / 180.0 / 3600
        rcore=self.theta_c * self.dl.value * np.pi / 180.0 / 3600
        return(2 * self.sigma0**2 / GG * rcut / (rcut-rcore) *
               (rcut*np.arctan(r / rcut) - rcore*np.arctan(r / rcore)))

    def vcirc(self,r):
        """
        Compute the circular velocity at radius r.

        Args:
            r (float): Radius.

        Returns:
            float: Circular velocity at r.
        """
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        return(np.sqrt(GG*self.m3Dr(r)/r))

    def rt(self):
        """
        Compute the truncation radius using the sigma-rt relation.

        Returns:
            float: Truncation radius.
        """
        # best fit sigma-rt relation from Bergamini et al. 2019
        r_t = 32.01 * (np.sqrt(3.0 / 2.0) * self.sigma0 / 350.0) ** 2.42
        return r_t

    def density(self,r_in):
        """
        Compute the 3D density at radius r_in.

        Args:
            r_in (float): Radius.

        Returns:
            float: 3D density at r_in.
        """
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        rcut = self.theta_t * self.dl.value * np.pi / 180.0 / 3600
        rcore = self.theta_c * self.dl.value * np.pi / 180.0 / 3600
        r = r_in * self.dl.value * np.pi / 180.0 / 3600
        den = self.sigma0**2.0 / 2.0 / np.pi / GG * (rcut + rcore) / (rcore ** 2 * rcut) / \
              (1.0 + (r/rcore)**2)/(1.0+(r/rcut)**2)
        return den

    def surf_density(self,r_in):
        """
        Compute the surface density at radius r_in.

        Args:
            r_in (float): Radius.

        Returns:
            float: Surface density at r_in.
        """
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        rcut = self.theta_t * self.dl.value * np.pi / 180.0 / 3600
        rcore = self.theta_c * self.dl.value * np.pi / 180.0 / 3600
        r = r_in * self.dl.value * np.pi / 180.0 / 3600
        den = self.sigma0**2.0 / 2.0 / GG * rcut/(rcut - rcore) * \
              (1.0/np.sqrt(rcore**2 + r**2)-1.0/np.sqrt(rcut**2+r**2))
        return den

    def fit2nfw(self, nl, rmin=1e-3, rmax=1000.0, **kwargs_fit):
        """
        Fit the parameters of the PIEMD model to a NFW profile.

        Args:
            nl: Input NFW model.
            rmin (float): Minimum radius for fitting in arcsec.
            rmax (float): Maximum radius for fitting in arcsec.
            **kwargs_fit: Dictionary with keys: sigma0_in, sigma0_min, sigma0_max, theta_c_in, theta_c_min, theta_c_max, theta_t_in, theta_t_min, theta_t_max.

        Returns:
            None
        """

        if 'sigma0_in' in kwargs_fit:
            sigma0_in = kwargs_fit['sigma0_in']
        else:
            sigma0_in = 1000.0

        if 'sigma0_min' in kwargs_fit:
            sigma0_min = kwargs_fit['sigma0_min']
        else:
            sigma0_min = 10.0

        if 'sigma0_max' in kwargs_fit:
            sigma0_max = kwargs_fit['sigma0_max']
        else:
            sigma0_max = 2000.0

        if 'theta_c_in' in kwargs_fit:
            theta_c_in = kwargs_fit['theta_c_in']
        else:
            theta_c_in = 0.05

        if 'theta_c_min' in kwargs_fit:
            theta_c_min = kwargs_fit['theta_c_min']
        else:
            theta_c_min = 0.001

        if 'theta_c_max' in kwargs_fit:
            theta_c_max = kwargs_fit['theta_c_max']
        else:
            theta_c_max = 200.0

        if 'theta_t_in' in kwargs_fit:
            theta_t_in = kwargs_fit['theta_t_in']
        else:
            theta_t_in = 1000.0

        if 'theta_t_min' in kwargs_fit:
            theta_t_min = kwargs_fit['theta_t_min']
        else:
            theta_t_min = 1.0

        if 'theta_t_max' in kwargs_fit:
            theta_t_max = kwargs_fit['theta_t_max']
        else:
            theta_t_max = 2000.0

        r = np.logspace(np.log10(rmin),np.log10(rmax),100)

        p = lmfit.Parameters()
        p.add_many(('sigma0', np.log10(sigma0_in), True, np.log10(sigma0_min), np.log10(sigma0_max)),
                   ('theta_c', np.log10(theta_c_in), True, np.log10(theta_c_min),  np.log10(theta_c_max)),
                   ('theta_t', np.log10(theta_t_in), True, np.log10(theta_t_min),  np.log10(theta_t_max)))


        def cost_function(p):
            v = p.valuesdict()
            self.sigma0 = 10 ** p['sigma0']
            self.theta_c = 10 ** p['theta_c']
            self.theta_p = 10 ** p['theta_t']
            cost=(np.log10(self.density(r))-np.log10(nl.density(r)))**2/np.log10(nl.density(r))**2
            return cost

        mini = lmfit.Minimizer(cost_function, p)
        mi = mini.minimize()  # residual, p, method='Nelder')
        self.sigma0 = 10 ** mi.params['sigma0']
        self.theta_c = 10 ** mi.params['theta_c']
        self.theta_t = 10 ** mi.params['theta_t']
