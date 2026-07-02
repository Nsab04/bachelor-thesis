from pyLensLib.genlen import genlen
import numpy as np

class deflector(genlen):
    """
    Represents a gravitational lens deflector, inheriting from genlen.

    Attributes:
        co: Cosmology object for distance calculations.
        angx, angy: Deflection angle maps.
        pot: Potential map.
        zl (float): Lens redshift.
        zs (float): Source redshift.
        dl, ds, dls (float): Angular diameter distances.
        a11, a12, a21, a22: Gradients of deflection angles.
        usePotential (bool): Whether to use potential for deflection calculation.
        computed_potential (bool): Whether potential was computed.
    """
    def __init__(self, co, angx=None, angy=None, pot=None, usePotential=False, **kwargs):
        """
        Initialize a deflector object.

        Args:
            co: Cosmology object for distance calculations.
            angx, angy: Deflection angle maps (optional).
            pot: Potential map (optional).
            usePotential (bool): If True, compute deflection from potential.
            zl (float): Lens redshift (optional, default 0.5).
            zs (float): Source redshift (optional, default 1.0).
        """
        super().__init__()
        self.usePotential = usePotential
        self.computed_potential = False
        self.zl = kwargs.get('zl', 0.5)
        self.zs = kwargs.get('zs', 1.0)

        # Choose the deflection angles:
        # - to be computed from the potential, or
        # - use directly the given values of the deflection angles
        if usePotential:
            self.computed_potential = True
            self.pot = pot
            self.angy, self.angx = np.gradient(self.pot)
        else:
            self.angx = angx
            self.angy = angy
            if pot is not None:
                self.pot = pot
                self.computed_potential = True
        self.co = co
        self.dl = self.co.angular_diameter_distance(self.zl)
        self.ds = self.co.angular_diameter_distance(self.zs)
        self.dls = self.co.angular_diameter_distance_z1z2(self.zl, self.zs)
        self.a21, self.a11 = np.gradient(self.angx)
        self.a22, self.a12 = np.gradient(self.angy)

    def angle(self, theta1, theta2):
        """
        Get the deflection angles for the given grid.

        Args:
            theta1, theta2 (np.ndarray): Grid coordinates.

        Returns:
            tuple: Deflection angle maps (angx, angy).
        """
        if theta1.shape != self.angx.shape or theta2.shape != self.angy.shape:
            raise Exception('Incompatible sizes of grid and deflection angle maps')
        dtheta = np.abs(theta1[0, 1] - theta1[0, 0])
        if self.usePotential:
            return (self.angx / dtheta, self.angy / dtheta)
        else:
            return (self.angx, self.angy)

    def kappa(self, theta1, theta2):
        """
        Compute the convergence (kappa) for the given grid.

        Args:
            theta1, theta2 (np.ndarray): Grid coordinates.

        Returns:
            np.ndarray: Convergence map.
        """
        if theta1.shape != self.angx.shape or theta2.shape != self.angy.shape:
            raise Exception('Incompatible sizes of grid and deflection angle maps')
        dtheta = np.abs(theta1[0, 1] - theta1[0, 0])
        if self.usePotential:
            return (0.5 * (self.a11 / dtheta ** 2 + self.a22 / dtheta ** 2))
        else:
            return (0.5 * (self.a11 / dtheta + self.a22 / dtheta))

    def gamma(self, theta1, theta2):
        """
        Compute the shear (gamma) for the given grid.

        Args:
            theta1, theta2 (np.ndarray): Grid coordinates.

        Returns:
            tuple: Shear components (gamma1, gamma2).
        """
        if theta1.shape != self.angx.shape or theta2.shape != self.angy.shape:
            raise Exception('Incompatible sizes of grid and deflection angle maps')
        dtheta = np.abs(theta1[0, 1] - theta1[0, 0])
        if self.usePotential:
            return (0.5 * (self.a11 - self.a22) / dtheta ** 2, self.a12 / dtheta ** 2)
        else:
            return (0.5 * (self.a11 - self.a22) / dtheta, self.a12 / dtheta)

