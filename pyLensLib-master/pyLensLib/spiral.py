import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_modifier(modifier):
    """Normalize a spiral modifier field to the range [0, 1]."""
    s_min = float(np.min(modifier))
    s_max = float(np.max(modifier))
    if np.isclose(s_min, s_max):
        return np.ones_like(modifier)
    return (modifier - s_min) / (s_max - s_min)


def spiral_modifier_from_polar(r, theta, A, Na, Phid, alpha, re):
    """Build the raw spiral perturbation from polar coordinates."""
    r_safe = np.maximum(r, 1.0e-12)
    re_safe = max(float(re), 1.0e-12)
    return 1.0 + A * np.cos(Na * theta + (alpha * np.log(2.0 * r_safe / re_safe) + Phid))


def spiral_modifier_from_grid(y1, y2, ys1, ys2, pa, phi, A, Na, Phid, alpha, re):
    """Build a normalized spiral modifier from a coordinate grid."""
    x = np.cos(pa) * (y1 - ys1) + np.sin(pa) * (y2 - ys2)
    y = (-np.sin(pa) * (y1 - ys1) + np.cos(pa) * (y2 - ys2)) / (np.cos(phi) + 1.0e-2)
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return normalize_modifier(spiral_modifier_from_polar(r, theta, A, Na, Phid, alpha, re))


class spiral(object):
    """
    Add a spiral structure to an exponential disc.
    Implements the model of Metcalf et al. (2019), Eq. 2.

    Attributes:
        se: Input Sersic model (exponential disc, face-on).
        A (float): Amplitude of spiral structure.
        Na (int): Number of spiral arms.
        Phid (float): Disc phase.
        alpha (float): Looping parameter.
        phi (float): Tilting angle (0=face-on, pi=edge-on).
        z0 (float): Smoothing scale for convolution.
    """

    def __init__(self, se, A, Na, Phid, alpha, phi, z0=0.05):
        """
        Initialize a spiral structure to be added to the Sersic model.

        Args:
            se: Input Sersic model (should be an exponential disc, face-on).
            A (float): Amplitude of spiral structure.
            Na (int): Number of spiral arms.
            Phid (float): Disc phase.
            alpha (float): Looping parameter.
            phi (float): Tilting angle (0=face-on, pi=edge-on).
            z0 (float, optional): Smoothing scale for convolution.

        Returns:
            None
        """
        self.se = se
        self.A = A
        self.Na = Na
        self.Phid = Phid
        self.alpha = alpha
        self.phi = phi
        self.z0 = z0

        # some checks on se
        if self.se.n != 1.0:
            print('WARNING: the disc should have an exponential behaviour')
        if self.se.q != 1.0:
            print('WARNING: The sersic model is not circular! Setting q=1')
            print('Use parameter phi to change the projected galaxy ellipticity')
            self.se.q = 1.0

    def phir(self, r):
        """
        Compute the modified phase at distance r.

        Args:
            r (float or array): Distance from center.

        Returns:
            float or array: Modified phase value.
        """
        return self.alpha * np.log(2 * np.maximum(r, 1.0e-12) / self.se.re) + self.Phid

    def spiral_modifier(self, r, theta):
        """
        Compute the multiplicative factor to add a spiral structure to the disc.

        Args:
            r (float or array): Distance from center.
            theta (float or array): Polar angle.

        Returns:
            float or array: Disc modifier value.
        """
        return spiral_modifier_from_polar(r, theta, self.A, self.Na, self.Phid, self.alpha, self.se.re)

    def brightness(self):
        """
        Add spiral structure to the smooth disc and return the image.

        Returns:
            ndarray: Image including spiral structure (conserving flux).
        """
        px = self.se.sizex / (self.se.Nx - 1)
        s = spiral_modifier_from_grid(
            self.se.y1,
            self.se.y2,
            self.se.ys1,
            self.se.ys2,
            self.se.pa,
            self.phi,
            self.A,
            self.Na,
            self.Phid,
            self.alpha,
            self.se.re,
        )

        se_image = self.se.sie() * np.exp(-self.se.bn * ((np.sqrt(
            (np.cos(self.se.pa) * (self.se.y1 - self.se.ys1) + np.sin(self.se.pa) * (self.se.y2 - self.se.ys2)) ** 2 +
            ((-np.sin(self.se.pa) * (self.se.y1 - self.se.ys1) + np.cos(self.se.pa) * (self.se.y2 - self.se.ys2)) /
             (np.cos(self.phi) + 1.e-2)) ** 2
        ) / self.se.re) ** (1.0 / self.se.n) - 1.0)) * px * px

        norm = se_image.sum()

        brightness = se_image * s
        brightness = brightness / np.sum(brightness) * norm

        brightness_convolved = gaussian_filter(brightness, self.z0 / px)

        return brightness_convolved