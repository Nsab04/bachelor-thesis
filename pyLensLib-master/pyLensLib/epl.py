from astropy import constants as const
from astropy import cosmology
import astropy.units as units
import numpy as np
from scipy.special import hyp2f1

from pyLensLib.genlen import genlen


class epl(genlen):
    """
    Elliptical Power-Law (EPL) mass model using the Tessore & Metcalf
    formulation (MNRAS, 2015, 453, 2577).

    Notes
    -----
    - Slope parameterization follows ``t = gamma - 1``.
    - ``theta_E`` is the circularized Einstein radius (arcsec), matching the
      convention used in lenstronomy's EPL profile.
    - For ``t = 1`` this reduces to the singular isothermal elliptical case.
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

        self.x1 = kwargs.get("x1", 0.0)
        self.x2 = kwargs.get("x2", 0.0)
        self.pa = kwargs.get("pa", 0.0)

        self.q = float(kwargs.get("q", 1.0))
        if self.q <= 0.0 or self.q > 1.0:
            raise ValueError("q must be in (0, 1].")

        # Optional central softening radius (arcsec) for numerical regularization.
        self.theta_c = float(kwargs.get("theta_c", 0.0))
        if self.theta_c < 0.0:
            raise ValueError("theta_c must be >= 0.")

        if "t" in kwargs:
            self.t = kwargs["t"]
        elif "gamma_pl" in kwargs:
            self.t = kwargs["gamma_pl"] - 1.0
        else:
            self.t = kwargs.get("gamma", 2.0) - 1.0

        self.dl = self.co.angular_diameter_distance(self.zl)
        self.ds = self.co.angular_diameter_distance(self.zs)
        self.dls = self.co.angular_diameter_distance_z1z2(self.zl, self.zs)

        # theta_E is circularized Einstein radius.
        if "theta_E" in kwargs:
            self.theta_E = float(kwargs["theta_E"])
        elif "b" in kwargs:
            self.theta_E = float(kwargs["b"]) / np.sqrt(self.q)
        elif "sigma0" in kwargs:
            sigma0 = float(kwargs["sigma0"])
            conv = 180.0 / np.pi * 3600.0
            self.theta_E = (
                conv
                * 4.0
                * np.pi
                * sigma0**2
                / const.c.to("km/s").value**2
                * self.dls.value
                / self.ds.value
            )
        else:
            self.theta_E = 1.0

        if self.theta_E <= 0.0:
            raise ValueError("theta_E must be > 0.")

        # Keep sigma0 for interface consistency with other models.
        thetae_rad = self.theta_E * np.pi / 180.0 / 3600.0
        self.sigma0 = np.sqrt(
            thetae_rad * const.c.to("km/s").value**2 / (4.0 * np.pi) * self.ds.value / self.dls.value
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
        self.t = float(value) - 1.0

    @property
    def b(self):
        """Major-axis normalization used internally by the Tessore-Metcalf form."""
        return self.theta_E * np.sqrt(self.q)

    @b.setter
    def b(self, value):
        value = float(value)
        if value <= 0.0:
            raise ValueError("b must be > 0.")
        self.theta_E = value / np.sqrt(self.q)

    def _coords_rot(self, theta1__, theta2__):
        theta1 = np.asarray(theta1__, dtype=float) - self.x1
        theta2 = np.asarray(theta2__, dtype=float) - self.x2
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        return theta1_, theta2_

    def _ell_radius(self, theta1_, theta2_):
        r2 = (self.q * theta1_) ** 2 + theta2_**2 + self.theta_c**2
        return np.sqrt(np.maximum(r2, np.finfo(float).eps))

    def _alpha_internal(self, theta1_, theta2_):
        """
        Deflection in the major-axis frame (Tessore & Metcalf 2015).
        """
        rr = self._ell_radius(theta1_, theta2_)
        zz = self.q * theta1_ + 1j * theta2_
        zzc = np.conjugate(zz)
        ratio = np.ones_like(zz, dtype=np.complex128)
        np.divide(zz, zzc, out=ratio, where=np.abs(zzc) > np.finfo(float).eps)

        flattening = (1.0 - self.q) / (1.0 + self.q)
        hyper = hyp2f1(1.0, 0.5 * self.t, 2.0 - 0.5 * self.t, -flattening * ratio)
        romega = zz * hyper

        alpha = 2.0 / (1.0 + self.q) * np.power(self.b / rr, self.t) * romega
        alpha = np.nan_to_num(alpha)
        return np.real(alpha), np.imag(alpha), rr

    def _hessian_internal(self, theta1_, theta2_):
        """
        Potential second derivatives in major-axis frame.
        """
        alpha_x, alpha_y, rr = self._alpha_internal(theta1_, theta2_)
        kappa = 0.5 * (2.0 - self.t) * np.power(self.b / rr, self.t)

        rad = np.sqrt(theta1_**2 + theta2_**2 + self.theta_c**2)
        rad = np.maximum(rad, np.finfo(float).eps)
        cosp = theta1_ / rad
        sinp = theta2_ / rad
        cos2p = 2.0 * cosp**2 - 1.0
        sin2p = 2.0 * sinp * cosp

        gamma1 = (1.0 - self.t) * (alpha_x * cosp - alpha_y * sinp) / rad - kappa * cos2p
        gamma2 = (1.0 - self.t) * (alpha_y * cosp + alpha_x * sinp) / rad - kappa * sin2p

        psi11 = kappa + gamma1
        psi22 = kappa - gamma1
        psi12 = gamma2
        return psi11, psi22, psi12

    def kappa(self, theta1__, theta2__):
        theta1_, theta2_ = self._coords_rot(theta1__, theta2__)
        rr = self._ell_radius(theta1_, theta2_)
        return 0.5 * (2.0 - self.t) * np.power(self.b / rr, self.t)

    def angle(self, theta1__, theta2__):
        theta1_, theta2_ = self._coords_rot(theta1__, theta2__)
        alpha_x, alpha_y, _ = self._alpha_internal(theta1_, theta2_)

        alpha1 = alpha_x * np.sin(self.pa) + alpha_y * np.cos(self.pa)
        alpha2 = -alpha_x * np.cos(self.pa) + alpha_y * np.sin(self.pa)
        return alpha1, alpha2

    def alpha(self, theta1__, theta2__):
        """Backward-compatible alias for :meth:`angle`."""
        return self.angle(theta1__, theta2__)

    def potential(self, theta1__, theta2__):
        theta1_, theta2_ = self._coords_rot(theta1__, theta2__)
        alpha_x, alpha_y, _ = self._alpha_internal(theta1_, theta2_)
        return (theta1_ * alpha_x + theta2_ * alpha_y) / (2.0 - self.t)

    def gamma(self, theta1__, theta2__):
        psi11, psi22, psi12 = self._hessian_internal(*self._coords_rot(theta1__, theta2__))

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

    def sigma_crit(self):
        """Critical surface density in Msun/Mpc^2."""
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
        rr = np.sqrt(np.asarray(r_in, dtype=float) ** 2 + self.theta_c**2)
        rr = np.maximum(rr, np.finfo(float).eps)
        kappa = 0.5 * (2.0 - self.t) * np.power(self.theta_E / rr, self.t)
        return kappa * self.sigma_crit()
