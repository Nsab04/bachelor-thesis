class perturber(object):
    """
    Class representing a lensing perturber/deflector.

    Attributes:
        pa1 (float): Position angle 1.
        pa2 (float): Position angle 2.
        pkappa (float): Convergence perturbation.
        pgamma1 (float): Shear perturbation component 1.
        pgamma2 (float): Shear perturbation component 2.
        pg1 (float): Gradient perturbation component 1.
        pg2 (float): Gradient perturbation component 2.
        pf1 (float): Flexion perturbation component 1.
        pf2 (float): Flexion perturbation component 2.
    """

    def __init__(self,**kwargs):
        """
        Initialize a perturber object with lensing perturbation parameters.

        Args:
            **kwargs: Keyword arguments for lensing parameters (pa1, pa2, pkappa, pgamma1, pgamma2, pg1, pg2, pf1, pf2).

        Returns:
            None
        """


        if ('pa1' in kwargs):
            self.pa1 = kwargs['pa1']
        else:
            self.pa1=0.0
        if ('pa2' in kwargs):
            self.pa2 = kwargs['pa2']
        else:
            self.pa2=0.0
        if ('pkappa' in kwargs):
            self.pkappa = kwargs['pkappa']
        else:
            self.pkappa=0.0
        if ('pgamma1' in kwargs):
            self.pgamma1 = kwargs['pgamma1']
        else:
            self.pgamma1=0.0
        if ('pgamma2' in kwargs):
            self.pgamma2 = kwargs['pgamma2']
        else:
            self.pgamma2=0.0
        if ('pg1' in kwargs):
            self.pg1 = kwargs['pg1']
        else:
            self.pg1=0.0
        if ('pg2' in kwargs):
            self.pg2 = kwargs['pg2']
        else:
            self.pg2=0.0
        if ('pf1' in kwargs):
            self.pf1 = kwargs['pf1']
        else:
            self.pf1=0.0
        if ('pf2' in kwargs):
            self.pf2 = kwargs['pf2']
        else:
            self.pf2=0.0

