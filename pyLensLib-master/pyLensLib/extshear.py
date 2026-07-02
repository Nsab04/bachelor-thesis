import numpy as np

class ext_shear(object):
    """
    Represents an external shear lensing effect.

    Attributes:
        g (float): Shear amplitude.
        phi_g (float): Shear direction in radians.
    """
    def __init__(self, g, phi_g):
        """
        Initialize an external shear.

        Args:
            g (float): Shear amplitude.
            phi_g (float): Shear direction in degrees.
        """
        self.g = g
        self.phi_g = np.deg2rad(phi_g)

    def psi(self, x, phi):
        """
        Compute the lensing potential at polar coordinates (x, phi).

        Args:
            x (float or np.ndarray): Distance from the center of the lens plane.
            phi (float or np.ndarray): Position angle.

        Returns:
            float or np.ndarray: Lensing potential value(s).
        """
        return 0.5 * self.g * x ** 2 * np.cos(2 * (phi - self.phi_g))


    def alpha(self, x, phi):
        """
        Compute the deflection angle at polar coordinates (x, phi).

        Args:
            x (float or np.ndarray): Distance from the center of the lens plane.
            phi (float or np.ndarray): Position angle.

        Returns:
            tuple: Deflection angle components (a1, a2).
        """
        a1 = self.g * x * np.cos(2 * self.phi_g - phi)
        a2 = self.g * x * np.sin(2 * self.phi_g - phi)
        return a1, a2

    def gamma(self):
        """
        Get the shear components (constant across the lens plane).

        Returns:
            tuple: Shear components (gamma1, gamma2).
        """
        g1 = self.g * np.cos(2 * self.phi_g)
        g2 = self.g * np.sin(2 * self.phi_g)
        return g1, g2