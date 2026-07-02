from astropy import constants as const
from astropy import cosmology
import numpy as np
import astropy.units as units
import uncertainties
from pyLensLib.genlen import genlen
from scipy.integrate import quad
import lmfit

tol = 1.0e-3

def recenter_and_rotate(x, y, x0, y0, phi, e):
    """
    Perform translation and rotation of the coordinates x, y.

    Args:
        x (float or array): Input x coordinates.
        y (float or array): Input y coordinates.
        x0 (float): x coordinate of the new origin.
        y0 (float): y coordinate of the new origin.
        phi (float): Rotation angle.
        e (float): Ellipticity (used to compute the elliptical radius).

    Returns:
        tuple: New x, y coordinates and elliptical radius (x_, y_, v).
    """
    x_ = x - x0
    y_ = y - y0
    x__ = x_ * np.cos(phi) + y_ * np.sin(phi)
    y__ = -x_ * np.sin(phi) + y_ * np.cos(phi)
    v  = np.sqrt((1 + e) * np.power(x__ , 2) + np.power(y__ , 2) * (1 - e))
    return x_, y_, v

def v1(e, pa, v, theta1, theta2):
    """
    Compute the first derivative of the elliptical radius v with respect to x.

    Args:
        e (float): Ellipticity.
        pa (float): Rotation angle.
        v (float or array): Elliptical radius.
        theta1 (float or array): x coordinate.
        theta2 (float or array): y coordinate.

    Returns:
        float or array: First derivative v1.
    """
    return (theta1 + e * (theta1 * np.cos(2 * pa) + theta2 * np.sin(2 * pa))) / v

def v2(e, pa, v, theta1, theta2):
    """
    Compute the first derivative of the elliptical radius v with respect to y.

    Args:
        e (float): Ellipticity.
        pa (float): Rotation angle.
        v (float or array): Elliptical radius.
        theta1 (float or array): x coordinate.
        theta2 (float or array): y coordinate.

    Returns:
        float or array: First derivative v2.
    """
    return (theta2 + e * (theta1 * np.sin(2 * pa) - theta2 * np.cos(2 * pa))) / v

def v11(e, pa, v, theta1, theta2):
    """
    Compute the second derivative of v1 with respect to x.

    Args:
        e (float): Ellipticity.
        pa (float): Rotation angle.
        v (float or array): Elliptical radius.
        theta1 (float or array): x coordinate.
        theta2 (float or array): y coordinate.

    Returns:
        float or array: Second derivative v11.
    """
    return (1.0 + e * np.cos(2 * pa) - v1(e,pa,v,theta1,theta2)**2) / v

def v22(e, pa, v, theta1, theta2):
    """
    Compute the second derivative of v2 with respect to y.

    Args:
        e (float): Ellipticity.
        pa (float): Rotation angle.
        v (float or array): Elliptical radius.
        theta1 (float or array): x coordinate.
        theta2 (float or array): y coordinate.

    Returns:
        float or array: Second derivative v22.
    """
    return (1 - e * np.cos(2 * pa) - v2(e,pa,v,theta1,theta2)**2) / v

def v12(e, pa, v, theta1, theta2):
    """
    Compute the second derivative of v1 with respect to x and y.

    Args:
        e (float): Ellipticity.
        pa (float): Rotation angle.
        v (float or array): Elliptical radius.
        theta1 (float or array): x coordinate.
        theta2 (float or array): y coordinate.

    Returns:
        float or array: Second derivative v12.
    """
    return (e * np.sin(2 * pa) - v1(e, pa, v, theta1, theta2)*
            v2(e, pa, v, theta1, theta2))/ v

def alpha_kernel(v,rs_as,tol=1e-5):
    """
    Compute the deflection angle of an NFW halo for an axially symmetric lens.

    Args:
        v (float or array): Radius.
        rs_as (float): Scale radius in arcsec.
        tol (float): Minimum radius (to avoid divergence at v=0).

    Returns:
        float or array: Deflection angle kernel at v.
    """
    x = np.abs(v / rs_as)

    fx = np.piecewise(x, [x < tol / 2, x > 1., x < tol, (x < 1.) & (x > tol), x == 1.],
                      [0.0,
                       lambda x: 1.0 / x * (np.log(x / 2.) + 2.0 / np.sqrt(x * x - 1.) *
                                            np.arctan(np.sqrt((x - 1.) / (x + 1.)))),
                       lambda x: -x * (2.0 * np.log(0.5 * x) + 1) / 4.0,
                       lambda x: 1.0 / x * (np.log(x / 2.) + 2.0 / np.sqrt(1. - x * x) *
                                            np.arctanh(np.sqrt((1. - x) / (1. + x)))),
                       (np.log(1. / 2.) + 1.)])
    return 4.0 * fx

def kappa_kernel(v,rs_as):
    """
    Compute the convergence of an NFW halo for an axially symmetric lens.

    Args:
        v (float or array): Radius.
        rs_as (float): Scale radius in arcsec.

    Returns:
        float or array: Convergence kernel at v.
    """
    x = np.abs(v / rs_as)
    fx = np.piecewise(x, [x > 1., x < 1., x == 1.],
                      [lambda x: 1.0 - 2.0 / np.sqrt(x * x - 1.) * np.arctan(np.sqrt((x - 1.)/(x + 1.))),
                       lambda x: 1.0 - 2.0 / np.sqrt(1. - x * x) * np.arctanh(np.sqrt((1. - x )/(1. + x))),
                       0.0])
    return 2.0 * fx/(x**2-1.0)

def pot_kernel(v,rs_as):
    """
    Compute the potential of an NFW halo for an axially symmetric lens.

    Args:
        v (float or array): Radius.
        rs_as (float): Scale radius in arcsec.

    Returns:
        float or array: Potential kernel at v.
    """
    x = np.abs(v / rs_as)
    fx = np.piecewise(x, [x > 1., x < 1., x == 1.],
                      [lambda x: np.log(x / 2.) ** 2
                                 + (1.0 / np.sqrt(x * x - 1.) * np.arctan(np.sqrt(x * x - 1.))) ** 2 * (x ** 2 - 1),
                       lambda x: np.log(x / 2.) ** 2 +
                                 (1.0 / np.sqrt(1. - x * x) * np.arctanh(np.sqrt(1. - x * x))) ** 2 * (1 - x ** 2),
                       np.log(1. / 2.) ** 2])
    return 2.0 * fx * rs_as ** 2

def integrand_mgnfw(r,a):
    """
    Integrand for the mass of a generalized NFW halo.

    Args:
        r (float): Radius.
        a (float): Slope parameter.

    Returns:
        float: Value of the integrand.
    """
    return r ** (2 - a) / (1 + r) ** (3 - a)

def integrand_agnfw(z, a, v):
    """
    Integrand for the deflection angle of a generalized NFW halo.

    Args:
        z (float): Integration variable.
        a (float): Slope parameter.
        v (float): Scaled radius.

    Returns:
        float: Value of the integrand.
    """
    return ((z + v)**(a - 3)) * ((1 - ((1 - z**2)**0.5)) / z)

class nfwell(genlen):

    def __init__(self, co=None, **kwargs):
        """
        Initialize a pseudo-elliptical NFW halo lensing model.

        This assumes an elliptical lensing potential, not an elliptical mass distribution.
        Only valid for small ellipticities.

        Args:
            co: Cosmology instance.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__()
        self.computed_potential = False
        if co is None:
            self.co = cosmology.FlatLambdaCDM(70, 0.3)
        else:
            self.co = co

        if 'x1' in kwargs:
            self.x1 = kwargs['x1']
        else:
            self.x1 = 0.5

        if 'x2' in kwargs:
            self.x2 = kwargs['x2']
        else:
            self.x2 = 0.5

        if 'pa' in kwargs:
            self.pa = kwargs['pa']
        else:
            self.pa = 0.0

        if 'mass' in kwargs:
            self.mass = kwargs['mass']
        else:
            self.mass = 1e15

        if 'zl' in kwargs:
            self.zl = kwargs['zl']
        else:
            self.zl = 0.5

        if 'zs' in kwargs:
            self.zs = kwargs['zs']
        else:
            self.zs = 2.0

        if 'q' in kwargs:
            self.q = kwargs['q']
        else:
            self.q = 1.0

        if 'conc' in kwargs:
            self.conc = kwargs['conc']
        else:
            self.conc, self.conc_nom = self.cmRel(cmtype='MM152D', cmkind='all')

        if 'a' in kwargs:
            self.a = kwargs['a']
        else:
            self.a = 1.0

        self.dl = self.co.angular_diameter_distance(self.zl)
        self.ds = self.co.angular_diameter_distance(self.zs)
        self.dls = self.co.angular_diameter_distance_z1z2(self.zl, self.zs)
        self.sc = self.sigma_crit()
        self.rs = self.R200() / self.conc
        self.rs_as = self.rs * self.co.arcsec_per_kpc_proper(self.zl).value * 1e3
        self.rhos = self.char_den()
        self.ks = self.rhos * self.rs / self.sc
        self.init_ad = False
        self.e = 1 - self.q
        self.prof_params = ['mass', 'conc']

    def angle(self, theta1__, theta2__):
        """
        Compute the reduced deflection angles at a given position.

        Args:
            theta1__ (float or array): First coordinate.
            theta2__ (float or array): Second coordinate.

        Returns:
            tuple: Deflection angle components (scaled to the lens and source redshifts).
        """
        theta1,theta2,v = recenter_and_rotate(theta1__, theta2__, self.x1, self.x2, self.pa, self.e)
        a1 = self.nfwalpha(v) * v1(self.e, self.pa, v, theta1, theta2)
        a2 = self.nfwalpha(v) * v2(self.e, self.pa, v, theta1, theta2)
        return a1, a2

    def potential(self, theta1__, theta2__):
        """
        Compute the lensing potential at a given position.

        Args:
            theta1__ (float or array): First coordinate.
            theta2__ (float or array): Second coordinate.

        Returns:
            float or array: Lensing potential (scaled to the lens and source redshifts).
        """
        theta1, theta2, v = recenter_and_rotate(theta1__, theta2__, self.x1, self.x2, self.pa, self.e)
        pot = self.nfwpot(v)
        return pot

    def nfwalpha(self, v):
        """
        Compute the NFW deflection angle for an axially symmetric NFW lens.

        Args:
            v (float or array): Elliptical radius.

        Returns:
            float or array: Modulus of the deflection angle at v.
        """
        fx = alpha_kernel(v, self.rs_as)
        alpha_e = self.ks * fx * self.rs_as
        return alpha_e



    def nfwkappa(self, v):
        """
        Compute the NFW convergence for an axially symmetric NFW lens.

        Args:
            v (float or array): Elliptical radius.

        Returns:
            float or array: Convergence at v.
        """
        fx = kappa_kernel(v, self.rs_as)
        kappa_e = self.ks * fx
        return kappa_e

    def nfwalphap(self, v):
        """
        Compute the first derivative of the NFW deflection angle for an axially symmetric NFW lens.

        Args:
            v (float or array): Elliptical radius.

        Returns:
            float or array: First derivative of deflection angle at v.
        """
        fx = alpha_kernel(v, self.rs_as)
        alpha_e = self.ks * fx * self.rs_as
        gx = kappa_kernel(v, self.rs_as)
        kappa_e = self.ks * gx
        return 2.0 * kappa_e - alpha_e/v

    def nfwpot(self, v):
        """
        Compute the NFW potential for an axially symmetric NFW lens.

        Args:
            v (float or array): Elliptical radius.

        Returns:
            float or array: Lensing potential at v.
        """
        fx = pot_kernel(v, self.rs_as)
        pot_e = self.ks * fx
        return pot_e

    def potential11(self, v, theta1, theta2):
        """
        Compute the second derivative with respect to theta1 of the NFW potential for an elliptical NFW lens.

        Args:
            v (float or array): Elliptical radius.
            theta1 (float or array): x coordinate on the lens plane.
            theta2 (float or array): y coordinate on the lens plane.

        Returns:
            float or array: Second derivative of the lensing potential with respect to theta1.
        """
        #theta1, theta2, v = recenter_and_rotate(theta1__, theta2__, self.x1, self.x2, self.pa, self.e)
        return (self.nfwalphap(v)*v1(self.e, self.pa, v, theta1, theta2)**2 +
                self.nfwalpha(v)*v11(self.e, self.pa, v, theta1, theta2))

    def potential22(self, v, theta1, theta2):
        """
        Compute the second derivative with respect to theta2 of the NFW potential for an elliptical NFW lens.

        Args:
            v (float or array): Elliptical radius.
            theta1 (float or array): x coordinate on the lens plane.
            theta2 (float or array): y coordinate on the lens plane.

        Returns:
            float or array: Second derivative of the lensing potential with respect to theta2.
        """
        #theta1, theta2, v = recenter_and_rotate(theta1__, theta2__, self.x1, self.x2, self.pa, self.e)
        return (self.nfwalphap(v)*v2(self.e, self.pa, v, theta1, theta2)**2 +
                self.nfwalpha(v)*v22(self.e, self.pa, v, theta1, theta2))

    def potential12(self, v, theta1, theta2):
        """
        Compute the second derivative with respect to theta1 and theta2 of the NFW potential for an elliptical NFW lens.

        Args:
            v (float or array): Elliptical radius.
            theta1 (float or array): x coordinate on the lens plane.
            theta2 (float or array): y coordinate on the lens plane.

        Returns:
            float or array: Second derivative of the lensing potential with respect to theta1 and theta2.
        """
        #theta1, theta2, v = recenter_and_rotate(theta1__, theta2__, self.x1, self.x2, self.pa, self.e)
        return (self.nfwalphap(v)*v1(self.e, self.pa, v, theta1, theta2)*v2(self.e, self.pa, v, theta1, theta2) +
                self.nfwalpha(v)*v12(self.e, self.pa, v, theta1, theta2))


    def sigma_crit(self):
        """
        Compute the critical surface mass density given the lens and source redshifts.

        Returns:
            float: Critical surface mass density.
        """
        c2G = (const.c ** 2 / const.G).to(units.Msun / units.Mpc)
        factor = c2G / (4 * np.pi)
        return (factor * (self.ds / (self.dl * self.dls))).value

    def R200(self):
        """
        Compute r200 using the mass and critical density.

        Returns:
            float: r200 value.
        """
        # r=1.63e-2*self.mass**(1./3.0)*(self.co.Om0/self.co.Om(self.zl))**(1./3.)/(1.0+self.zl)
        r = (self.mass / (4.0 / 3.0 * np.pi * 200 * self.co.critical_density(self.zl).to(
            units.Msun / units.Mpc ** 3).value)) ** (1. / 3.)

        return (r)

    def char_den(self):
        """
        Return the characteristic density of the halo.

        Returns:
            float: Characteristic density (rhos).
        """
        rhos = 200. / 3. * self.co.critical_density(self.zl).to(units.Msun / units.Mpc ** 3).value * \
               self.conc ** 3 / (np.log(1 + self.conc) - self.conc / (1.0 + self.conc))
        return rhos

    def density(self, r):
        """
        Compute the NFW density at a given radius.

        Args:
            r (float or array): Radius at which to compute the density.

        Returns:
            float or array: NFW density at the given radius.
        """
        return self.rhos / (r / self.rs_as) / (1.0 + r / self.rs_as) ** 2

    # def init_jacobian(self, theta1__, theta2__):
    #      """
    #      computes the Jacobian of the deflection angle at a given position
    #      :param theta1__: first coordinate
    #      :param theta2__: second coordinate
    #      :return: jacobian elements at each position
    #      """
    #      a1, a2 = self.angle(theta1__, theta2__)
    #      pixel = (theta1__.max() - theta1__.min()) / len(theta1__)
    #      self.a12, self.a11 = np.gradient(a1,edge_order=2) / pixel
    #      self.a22, self.a21 = np.gradient(a2,edge_order=2) / pixel
    #      self.init_ad = True

    def init_jacobian(self, theta1__, theta2__):
        """
        Compute the Jacobian of the deflection angle at a given position.

        Args:
            theta1__ (float or array): First coordinate.
            theta2__ (float or array): Second coordinate.

        Returns:
            None
        """
        theta1, theta2, v = recenter_and_rotate(theta1__, theta2__, self.x1, self.x2, self.pa, self.e)
        self.a11 = self.potential11(v, theta1, theta2)
        self.a22 = self.potential22(v, theta1, theta2)
        self.a21 = self.potential12(v, theta1, theta2)
        self.a12 = self.a21
        self.init_ad = True

    def kappa(self, theta1__, theta2__):
        """
        Compute the convergence at a given position using the Jacobian of the deflection angle.

        Args:
            theta1__ (float or array): First coordinate.
            theta2__ (float or array): Second coordinate.

        Returns:
            float or array: Convergence at each position.
        """
        if not self.init_ad:
            self.init_jacobian(theta1__, theta2__)
        return 0.5 * (self.a11 + self.a22)

    def kappa_exact(self,theta1__,theta2__):
        """
        Compute the convergence at a given position using the NFW formula.

        Args:
            theta1__ (float or array): First coordinate.
            theta2__ (float or array): Second coordinate.

        Returns:
            float or array: Convergence at each position.
        """
        theta1, theta2, v = recenter_and_rotate(theta1__, theta2__, self.x1, self.x2, self.pa, self.e)
        kappa = 0.5*(self.potential11(v, theta1, theta2) + self.potential22(v, theta1, theta2))
        return kappa

    def gamma_exact(self,theta1__,theta2__):
        """
        Compute the shear at a given position using the NFW formula.

        Args:
            theta1__ (float or array): First coordinate.
            theta2__ (float or array): Second coordinate.

        Returns:
            tuple: Components of the shear at each position (gamma1, gamma2).
        """
        theta1, theta2, v = recenter_and_rotate(theta1__, theta2__, self.x1, self.x2, self.pa, self.e)
        gamma1 = 0.5*(self.potential11(v, theta1, theta2) - self.potential22(v, theta1, theta2))
        gamma2 = self.potential12(v, theta1, theta2)
        return gamma1,gamma2

    def gamma(self, theta1__, theta2__):
        """
        Compute the shear at a given position using the Jacobian of the deflection angle.

        Args:
            theta1__ (float or array): First coordinate.
            theta2__ (float or array): Second coordinate.

        Returns:
            tuple: Components of the shear at each position (gamma1, gamma2).
        """
        if not self.init_ad:
            self.init_jacobian(theta1__, theta2__)
        return 0.5 * (self.a11 - self.a22), self.a21

    def cmRel(self, cmtype='MM152D', cmkind='all'):
        """
        Compute the concentration-mass relation for the halo.

        Args:
            cmtype (str): Type of concentration-mass relation.
            cmkind (str): Kind of relation ('all', 'rel', etc.).

        Returns:
            tuple: Concentration and concentration with uncertainty.
        """

        A = 0.0
        B = 0.0
        C = 0.0

        if (cmtype == 'MM152D'):
            if (cmkind == 'all'):
                A = 3.58
                B = 0.108
                C = -0.032
                # A = uncertainties.ufloat(3.58, 0.04)
                # B = uncertainties.ufloat(0.003, 0.053)
                # C = uncertainties.ufloat(0.051, 0.013)
            elif (cmkind == 'rel'):
                A = uncertainties.ufloat(3.813, 0.05)
                B = uncertainties.ufloat(0.108, 0.064)
                C = uncertainties.ufloat(-0.032, 0.015)
                # cov=[[0.04,0.0,0.0],[0.0,0.064,0.0],[0.0,0.0,0.013]]
            else:
                A = uncertainties.ufloat(4.38, 0.113)
                B = uncertainties.ufloat(0.420, 0.137)
                C = uncertainties.ufloat(-0.052, 0.030)

        conc_u = A * (1.34 / (1. + self.zl)) ** B * (self.mass / 8e14) ** C

        # conc = conc_u.nominal_value + np.random.normal(0.0, conc_u.std_dev, 1) + np.random.normal(0.0, 0.15, 1)
        # conc = conc_u.nominal_value + np.random.lognormal(0.0, 0.15, 1)
        conc = conc_u * np.random.lognormal(mean=0.0, sigma=0.38, size=1)[0]
        # return (conc[0])
        # print (conc, conc_u)
        return conc, conc_u  # (A * (1.34 / (1. + self.zl)) ** B * (self.mass / 8e14) ** C)

    def vmax(self):
        """
        Compute the maximum circular velocity for the halo.

        Returns:
            float: Maximum circular velocity (vmax).
        """
        return (1.64 * self.rs * np.sqrt(
            self.rhos * const.G.to(units.km ** 2 * units.Mpc / units.Msun / units.s ** 2))).value

    # functions for generalized NFW halos
    def m_gnfw(self, v):
        """
        Compute the mass for a generalized NFW halo.

        Args:
            v (float): Scaled radius.

        Returns:
            float: Mass within radius v.
        """
        integral, _ = quad(integrand_mgnfw, 0, v, args=(self.a,))
        return integral

    def gnfwalpha(self, v):
        """
        Compute the deflection angle for a generalized NFW halo.

        Args:
            v (float or array): Scaled radius.

        Returns:
            float or array: Deflection angle at each radius.
        """
        # Ensure v is an array
        v = np.atleast_1d(v)

        result = np.zeros_like(v)
        for i, val in enumerate(v.flatten()):
            integral, _ = quad(integrand_agnfw, 0, 1, args=(self.a, val))
            alpha = self.m_gnfw(val) / val + integral
            result.flat[i] = alpha
        return result.reshape(v.shape)

    def fit2piemd(self, piemd_model, rmin=1e-3, rmax=1000.0, **kwargs_fit):
        """
        Fit the parameters of the NFW model to a PIEMD profile.

        Args:
            piemd_model: Input PIEMD model.
            rmin (float): Minimum radius for fitting in arcsec.
            rmax (float): Maximum radius for fitting in arcsec.
            **kwargs_fit: Dictionary with keys: rs_in, rs_min, rs_max, rhos_in, rhos_min, rhos_max.

        Returns:
            None
        """

        if 'rs_in' in kwargs_fit:
            rs_in = kwargs_fit['rs_in']
        else:
            rs_in = 1000.0

        if 'rs_min' in kwargs_fit:
            rs_min = kwargs_fit['rs_min']
        else:
            rs_min = 10.0

        if 'rs_max' in kwargs_fit:
            rs_max = kwargs_fit['rs_max']
        else:
            rs_max = 2000.0

        if 'rhos_in' in kwargs_fit:
            rhos_in = kwargs_fit['rhos_in']
        else:
            rhos_in = 0.1

        if 'rhos_min' in kwargs_fit:
            rhos_min = kwargs_fit['rhos_min']
        else:
            rhos_min = 0.001

        if 'rhos_max' in kwargs_fit:
            rhos_max = kwargs_fit['rhos_max']
        else:
            rhos_max = 10.0

        r = np.logspace(np.log10(rmin), np.log10(rmax), 100)

        p = lmfit.Parameters()
        p.add_many(('rs', np.log10(rs_in), True, np.log10(rs_min), np.log10(rs_max)),
                   ('rhos', np.log10(rhos_in), True, np.log10(rhos_min), np.log10(rhos_max)))

        def cost_function(p):
            v = p.valuesdict()
            self.rs = 10 ** p['rs']
            self.rhos = 10 ** p['rhos']
            cost = (np.log10(self.density(r)) - np.log10(piemd_model.density(r))) ** 2 / np.log10(
                piemd_model.density(r)) ** 2
            return cost

        mini = lmfit.Minimizer(cost_function, p)
        mi = mini.minimize()
        self.rs = 10 ** mi.params['rs']
        self.rhos = 10 ** mi.params['rhos']
