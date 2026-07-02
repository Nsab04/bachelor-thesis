from astropy import constants as const
from astropy import cosmology
import astropy.units as units
import numpy as np

from pyLensLib.genlen import genlen


class pepl(genlen):
    """
    Pseudo Elliptical Power-Law (pEPL) lens model.

    This is the original pseudo-elliptical implementation used in pyLensLib.
    It follows the same interface philosophy as the other analytical
    lens models in pyLensLib: it provides convergence, deflection, shear, and
    potential on demand and can be used directly with genlen.setGrid().

    Notes
    -----
    - The mass-profile slope is parametrized through ``t`` (projected slope), with
      ``gamma_pl = t + 1`` (3D-like slope notation often used in the literature).
    - Isothermal corresponds to ``t = 1`` (``gamma = 2``).
    - ``theta_E`` is the angular scale in arcsec used for normalization.
    """

    def __init__(self, co=None, **kwargs):
        super().__init__()
        self.computed_potential = False

        if co is None:
            self.co = cosmology.FlatLambdaCDM(70.0, 0.3)
        else:
            self.co = co

        self.zl = kwargs.get("zl", 0.3)
        self.zs = kwargs.get("zs", 1.5)
        if self.zs <= self.zl:
            raise ValueError("EPL requires zs > zl.")

        # Geometrical parameters
        self.x1 = kwargs.get("x1", 0.0)
        self.x2 = kwargs.get("x2", 0.0)
        self.pa = kwargs.get("pa", 0.0)
        self.q = kwargs.get("q", 1.0)
        if self.q <= 0.0 or self.q > 1.0:
            raise ValueError("q must be in (0, 1].")

        # Inner softening scale in arcsec to regularize the center
        self.theta_c = kwargs.get("theta_c", 0.0)

        # Profile slope: prefer explicit t, otherwise derive from gamma-like input.
        if "t" in kwargs:
            self.t = kwargs["t"]
        elif "gamma_pl" in kwargs:
            self.t = kwargs["gamma_pl"] - 1.0
        else:
            self.t = kwargs.get("gamma", 2.0) - 1.0
        self._validate_t(self.t)

        # Distances
        self.dl = self.co.angular_diameter_distance(self.zl)
        self.ds = self.co.angular_diameter_distance(self.zs)
        self.dls = self.co.angular_diameter_distance_z1z2(self.zl, self.zs)

        # Normalization: either theta_E directly, or derive it from sigma0 using
        # the isothermal conversion for convenience.
        if "theta_E" in kwargs:
            self.theta_E = kwargs["theta_E"]
        elif "b" in kwargs:
            self.theta_E = kwargs["b"]
        elif "sigma0" in kwargs:
            sigma0 = kwargs["sigma0"]
            conv = 180.0 / np.pi * 3600.0
            self.theta_E = (
                conv
                * 4.0
                * np.pi
                * sigma0**2
                / const.c.to("km/s").value**2
                * self.dls.value
                / self.ds.value
                / np.sqrt(self.q)
            )
        else:
            self.theta_E = 1.0
        if self.theta_E <= 0.0:
            raise ValueError("theta_E must be > 0.")

        # Keep an equivalent sigma0 value for convenience/output consistency.
        thetae_rad = self.theta_E * np.pi / 180.0 / 3600.0
        self.sigma0 = np.sqrt(
            thetae_rad
            * const.c.to("km/s").value**2
            / (4.0 * np.pi)
            * self.ds.value
            / self.dls.value
            * np.sqrt(self.q)
        )

        self.prof_params = ["theta_E", "t"]

    @staticmethod
    def _validate_t(t):
        t = float(t)
        if t <= 0.0 or t >= 2.0:
            raise ValueError("t must be in (0, 2) for a physical EPL profile.")

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        value = float(value)
        self._validate_t(value)
        self._t = value

    @property
    def gamma_pl(self):
        return self.t + 1.0

    @gamma_pl.setter
    def gamma_pl(self, value):
        t = value - 1.0
        self._validate_t(t)
        self.t = t

    def _coords_rot(self, theta1__, theta2__):
        theta1 = theta1__ - self.x1
        theta2 = theta2__ - self.x2
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        return theta1_, theta2_

    def _core_ell_radius(self, theta1_, theta2_):
        r2 = self.q * theta1_**2 + theta2_**2 / self.q + self.theta_c**2
        r2 = np.maximum(r2, np.finfo(float).eps)
        return np.sqrt(r2), r2

    def _internal_derivatives(self, theta1__, theta2__):
        """
        Return internal-frame second derivatives of the potential.
        """
        theta1_, theta2_ = self._coords_rot(theta1__, theta2__)
        r, r2 = self._core_ell_radius(theta1_, theta2_)
        amp = self.theta_E**self.t * np.power(r, -self.t)

        psi11_ = amp * self.q - self.t * amp * self.q * self.q * theta1_**2 / r2
        psi22_ = amp / self.q - self.t * amp * theta2_**2 / (self.q * self.q * r2)
        psi12_ = -self.t * amp * theta1_ * theta2_ / r2
        return psi11_, psi22_, psi12_, theta1_, theta2_, r

    def kappa(self, theta1__, theta2__):
        """
        Convergence map of the EPL model.
        """
        psi11_, psi22_, _, _, _, _ = self._internal_derivatives(theta1__, theta2__)
        # Trace is invariant under rotation.
        return 0.5 * (psi11_ + psi22_)

    def angle(self, theta1__, theta2__):
        """
        Deflection angle components (alpha1, alpha2) in arcsec.
        """
        theta1_, theta2_ = self._coords_rot(theta1__, theta2__)
        r, _ = self._core_ell_radius(theta1_, theta2_)
        amp = self.theta_E**self.t * np.power(r, -self.t)

        alphax = amp * self.q * theta1_
        alphay = amp * theta2_ / self.q

        alphax_ = alphax * np.sin(self.pa) + alphay * np.cos(self.pa)
        alphay_ = -alphax * np.cos(self.pa) + alphay * np.sin(self.pa)
        return alphax_, alphay_

    def alpha(self, theta1__, theta2__):
        """
        Backward-compatible alias of :meth:`angle`.
        """
        return self.angle(theta1__, theta2__)

    def gamma(self, theta1__, theta2__):
        """
        Shear components (gamma1, gamma2).
        """
        psi11, psi22, psi12, _, _, _ = self._internal_derivatives(theta1__, theta2__)

        psi11_ = (
            psi11 * np.sin(self.pa) ** 2
            + 2.0 * psi12 * np.sin(self.pa) * np.cos(self.pa)
            + psi22 * np.cos(self.pa) ** 2
        )
        psi22_ = (
            psi11 * np.cos(self.pa) ** 2
            - 2.0 * psi12 * np.sin(self.pa) * np.cos(self.pa)
            + psi22 * np.sin(self.pa) ** 2
        )
        psi12_ = (
            -psi11 * np.sin(self.pa) * np.cos(self.pa)
            + psi12 * (np.sin(self.pa) ** 2 - np.cos(self.pa) ** 2)
            + psi22 * np.sin(self.pa) * np.cos(self.pa)
        )

        gamma1 = 0.5 * (psi11_ - psi22_)
        gamma2 = psi12_
        return gamma1, gamma2

    def potential(self, theta1__, theta2__):
        """
        EPL lensing potential.
        """
        theta1_, theta2_ = self._coords_rot(theta1__, theta2__)
        r, _ = self._core_ell_radius(theta1_, theta2_)
        if np.isclose(2.0 - self.t, 0.0):
            return self.theta_E**2 * np.log(r / self.theta_E)
        return self.theta_E**self.t / (2.0 - self.t) * np.power(r, 2.0 - self.t)

    def sigma_crit(self):
        """
        Critical surface density in Msun/Mpc^2.
        """
        return (
            const.c**2
            / (4.0 * np.pi * const.G)
            * self.ds
            / (self.dl * self.dls)
        ).to(units.Msun / units.Mpc**2).value

    def surf_density(self, r_in):
        """
        Circularized projected surface density profile (Msun/Mpc^2).
        """
        r = np.sqrt(np.asarray(r_in) ** 2 + self.theta_c**2)
        kappa = 0.5 * (2.0 - self.t) * np.power(self.theta_E / r, self.t)
        return kappa * self.sigma_crit()
