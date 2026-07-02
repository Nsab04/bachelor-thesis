from astropy.cosmology import FlatLambdaCDM
import astropy.constants as const
import astropy.units as units
from astropy import cosmology
# from numpy.core._multiarray_umath import ndarray

import numpy as np
from pyLensLib.genlen import genlen


class sie(genlen):
    """
    Singular Isothermal Ellipsoid (SIE) lens model for gravitational lensing.

    Attributes:
        zl (float): Lens redshift.
        zs (float): Source redshift.
        theta_c (float): Core radius.
        pa (float): Position angle (radians).
        q (float): Axis ratio.
        sigma0 (float): Velocity dispersion.
        x1, x2 (float): Lens center coordinates.
        co: Cosmology object.
        dl, ds, dls: Angular diameter distances.
        prof_params (list): Profile parameters.
    """

    def __init__(self, co, **kwargs):
        """
        Initialize the SIE lens model.

        Args:
            co: Cosmology object.
            **kwargs: Lens and model parameters.

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

        self.co = co
        self.dl = co.angular_diameter_distance(self.zl)
        self.ds = co.angular_diameter_distance(self.zs)
        self.dls = co.angular_diameter_distance_z1z2(self.zl, self.zs)
        self.prof_params = ['theta_c','sigma0']

    # @property
    def bsie(self):
        """
        Calculate the Einstein radius for the SIE model.

        Returns:
            float: Einstein radius in arcseconds.
        """
        conv = 180.0 / np.pi * 3600.0  # radians to arcsec
        return conv * 4.0 * np.pi * self.sigma0 ** 2 / const.c.to('km/s').value ** 2 * self.dls.value / self.ds.value / np.sqrt(
            self.q)

    def sfunc(self):
        """
        Calculate the core radius divided by sqrt(q).

        Returns:
            float: Scaled core radius.
        """
        return (self.theta_c / np.sqrt(self.q))


    def kappa(self, theta1__, theta2__):
        """
        Compute the convergence (kappa) at given coordinates.

        Args:
            theta1__, theta2__ (float or array): Coordinates in arcseconds.

        Returns:
            float or array: Convergence value.
        """
        theta1 = (theta1__ - self.x1)
        theta2 = (theta2__ - self.x2)
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        kappa_ = self.bsie() / 2 * (1.0 / np.sqrt(self.sfunc() ** 2 + theta1_ ** 2 + theta2_ ** 2 / self.q ** 2))
        return kappa_


    def angle(self, theta1__, theta2__):
        """
        Compute the deflection angle at given coordinates.

        Args:
            theta1__, theta2__ (float or array): Coordinates in arcseconds.

        Returns:
            tuple: Deflection angles (alphax, alphay).
        """
        theta1 = theta1__ - self.x1
        theta2 = theta2__ - self.x2
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        if (self.q < 1):
            alphax = self.bsie() * \
                     self.q / np.sqrt(1 - self.q ** 2) * \
                     np.arctan(np.sqrt(1 - self.q ** 2) * theta1_ / (psi + self.sfunc()))
            alphay = self.bsie() * \
                     self.q / np.sqrt(1 - self.q ** 2) * \
                     np.arctanh(np.sqrt(1 - self.q ** 2) * theta2_ / (psi + self.q ** 2 * self.sfunc()))
        elif (self.q == 1):
            alphax = self.bsie() * theta1_ / (psi + self.sfunc())
            alphay = self.bsie() * theta2_ / (psi + self.sfunc())
        else:
            print('q cannot be larger than 1')

        alphax_ = alphax * np.sin(self.pa) + alphay * np.cos(self.pa)
        alphay_ = -alphax * np.cos(self.pa) + alphay * np.sin(self.pa)
        return (alphax_, alphay_)


    def potential(self, theta1__, theta2__):
        """
        Compute the lensing potential at given coordinates.

        Args:
            theta1__, theta2__ (float or array): Coordinates in arcseconds.

        Returns:
            float or array: Lensing potential value.
        """
        theta1 = theta1__ - self.x1
        theta2 = theta2__ - self.x2
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        if (self.q < 1):
            alphax = self.bsie() * \
                     self.q / np.sqrt(1 - self.q ** 2) * \
                     np.arctan(np.sqrt(1 - self.q ** 2) * theta1_ / (psi + self.sfunc()))
            alphay = self.bsie() * \
                     self.q / np.sqrt(1 - self.q ** 2) * \
                     np.arctanh(np.sqrt(1 - self.q ** 2) * theta2_ / (psi + self.q ** 2 * self.sfunc()))
        elif (self.q == 1):
            alphax = self.bsie() * theta1_ / (psi + self.sfunc())
            alphay = self.bsie() * theta2_ / (psi + self.sfunc())
        else:
            print('q cannot be larger than 1')
        pot = theta1_*alphax+ \
              theta2_*alphay+ \
              self.bsie()*self.q*self.sfunc()*np.log((1.+self.q)*self.sfunc()/
                                                     np.sqrt((psi+self.sfunc())**2+(1.-self.q**2)*
                                                             theta1_**2))
        return pot


    def psi11(self, theta1_, theta2_):
        """
        Compute the second derivative psi11 of the lensing potential.

        Args:
            theta1_, theta2_ (float or array): Rotated coordinates.

        Returns:
            float or array: psi11 value.
        """
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        den = (1.0 + self.q ** 2) * self.sfunc() ** 2 + 2.0 * psi * self.sfunc() + theta1_ ** 2 + theta2_ ** 2
        psi11_ = self.bsie() * self.q / psi * (
                self.q ** 2 * self.sfunc() ** 2 + theta2_ ** 2 + self.sfunc() * psi) / den
        return (psi11_)


    def psi22(self, theta1_, theta2_):
        """
        Compute the second derivative psi22 of the lensing potential.

        Args:
            theta1_, theta2_ (float or array): Rotated coordinates.

        Returns:
            float or array: psi22 value.
        """
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        den = (1.0 + self.q ** 2) * self.sfunc() ** 2 + 2.0 * psi * self.sfunc() + theta1_ ** 2 + theta2_ ** 2
        psi22_ = self.bsie() * self.q / psi * (self.sfunc() ** 2 + theta1_ ** 2 + self.sfunc() * psi) / den
        return (psi22_)


    def psi12(self, theta1_, theta2_):
        """
        Compute the mixed second derivative psi12 of the lensing potential.

        Args:
            theta1_, theta2_ (float or array): Rotated coordinates.

        Returns:
            float or array: psi12 value.
        """
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        den = (1.0 + self.q ** 2) * self.sfunc() ** 2 + 2.0 * psi * self.sfunc() + theta1_ ** 2 + theta2_ ** 2
        psi12_ = -self.bsie() * self.q / psi * (theta1_ * theta2_) / den
        return (psi12_)


    def gamma(self, theta1__, theta2__):
        """
        Compute the shear (gamma) at given coordinates.

        Args:
            theta1__, theta2__ (float or array): Coordinates in arcseconds.

        Returns:
            tuple: Shear components (gammax, gammay).
        """
        theta1 = theta1__ - self.x1
        theta2 = theta2__ - self.x2
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        psi11 = self.psi11(theta1_, theta2_)
        psi22 = self.psi22(theta1_, theta2_)
        psi12 = self.psi12(theta1_, theta2_)
        psi11_ = psi11 * np.sin(self.pa) ** 2 + 2.0 * psi12 * np.sin(self.pa) * np.cos(self.pa) + psi22 * np.cos(
            self.pa) ** 2
        psi22_ = psi11 * np.cos(self.pa) ** 2 - 2.0 * psi12 * np.sin(self.pa) * np.cos(self.pa) + \
                 psi22 * np.sin(self.pa) ** 2
        psi12_ = -psi11 * np.sin(self.pa) * np.cos(self.pa) + psi12 * (np.sin(self.pa) ** 2 - np.cos(self.pa) ** 2) + \
                 psi22 * np.sin(self.pa) * np.cos(self.pa)
        # gammax = 0.5 * (self.psi11(theta1_, theta2_) - self.psi22(theta1_, theta2_))
        # gammay = self.psi12(theta1_, theta2_)
        gammax = 0.5 * (psi11_ - psi22_)
        gammay = psi12_

        # gammax_=gammax*np.cos(self.pa)-gammay*np.sin(self.pa)
        # gammay_ = gammax * np.sin(self.pa) + gammay * np.cos(self.pa)
        return (gammax, gammay)

    def density(self,r_in):
        """
        Compute the 3D density profile at a given radius.

        Args:
            r_in (float): Radius in arcseconds.

        Returns:
            float: Density value in physical units.
        """
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        rc = self.theta_c* self.dl.value * np.pi / 180.0 / 3600
        r = r_in * self.dl.value * np.pi / 180.0 / 3600
        den = self.sigma0**2.0 / 2.0 / np.pi / GG / (rc**2 + r**2)
        return den
