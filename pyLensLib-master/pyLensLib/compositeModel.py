"""
This module contains the composite model class that is used to create a composite model of multiple lens models.
"""

import numpy as np
import lmfit
from pyLensLib.genlen import genlen
from astropy import cosmology

class compositeModel(genlen):
    """
    This class creates a composite lens model from a list of lens models.
    """
    def __init__(self,co=None,models=None,**kwargs):
        """
        Constructor of the composite lens model.
        :param co: astropy cosmology object
        :param kwargs: keyword arguments
        """
        super().__init__()
        self.computed_potential = False
        if co is None:
            self.co = cosmology.FlatLambdaCDM(70, 0.3)
        else:
            self.co = co
        if models is None:
            self.models = []
        else:
            self.models = models

    def kappa(self, theta1, theta2):
        """
        Compute the convergence of the composite lens model.
        :param theta1: x-coordinate
        :param theta2: y-coordinate
        :return: convergence
        """
        kappa = np.zeros_like(theta1)
        for model in self.models:
            kappa += model.kappa(theta1, theta2)
        return kappa

    def gamma(self, theta1, theta2):
        """
        Compute the shear of the composite lens model.
        :param theta1: x-coordinate
        :param theta2: y-coordinate
        :return: shear
        """
        gamma1 = np.zeros_like(theta1)
        gamma2 = np.zeros_like(theta2)
        for model in self.models:
            g1, g2 = model.gamma(theta1, theta2)
            gamma1 += g1
            gamma2 += g2
        return gamma1, gamma2

    def angle(self, theta1, theta2):
        """
        Compute the deflection angle of the composite lens model.
        :param theta1: x-coordinate
        :param theta2: y-coordinate
        :return: deflection angle
        """
        alpha1 = np.zeros_like(theta1)
        alpha2 = np.zeros_like(theta2)
        for model in self.models:
            if hasattr(model, "angle"):
                a1, a2 = model.angle(theta1, theta2)
            elif hasattr(model, "alpha"):
                a1, a2 = model.alpha(theta1, theta2)
            else:
                raise AttributeError("Sub-model must implement 'angle(theta1, theta2)' or 'alpha(theta1, theta2)'")
            alpha1 += a1
            alpha2 += a2
        return alpha1, alpha2

    def alpha(self, theta1, theta2):
        """
        Backward-compatible alias for angle().
        """
        return self.angle(theta1, theta2)

    def potential(self, theta1, theta2):
        """
        Compute the potential of the composite lens model.
        :param theta1: x-coordinate
        :param theta2: y-coordinate
        :return: potential
        """
        pot = np.zeros_like(theta1)
        for model in self.models:
            pot += model.potential(theta1, theta2)
        return pot

    def density(self,r):
        """
        Compute the density of the composite lens model.
        :param r: radius
        :return: density
        """
        den = np.zeros_like(r)
        for model in self.models:
            den += model.density(r)
        return den

    def surf_density(self,r):
        """
        Compute the surface density of the composite lens model.
        :param r: radius
        :return: surface density
        """
        den = np.zeros_like(r)
        for model in self.models:
            den += model.surf_density(r)
        return den

    def m2Dr(self,r):
        """
        Compute the 2D mass distribution of the composite lens model.
        :param r: radius
        :return: mass distribution
        """
        mass = np.zeros_like(r)
        for model in self.models:
            mass += model.m2Dr(r)
        return mass

    def fit2piemd(self, input_model, rmin=1e-3, rmax=1000.0, **kwargs):
        """
        Fit the parameters of the composite lens model to an input density model.
        :param piemd_model: piemd model
        :param rmin: minimum radius for fitting in arcsec
        :param rmax: maximum radius for fitting in arcsec
        :param kwargs: parameters to be fit and flat priors
        :return:
        """
        if input_model is None:
            raise ValueError("input_model must be provided")
        if len(self.models) == 0:
            raise ValueError("No models available in compositeModel")

        r = np.logspace(np.log10(rmin), np.log10(rmax), 100)
        params = lmfit.Parameters()
        param_refs = []

        for i, model in enumerate(self.models):
            prof_params = getattr(model, "prof_params", [])
            for par in prof_params:
                default_value = getattr(model, par, None)
                param_name = f"m{i}_{par}"
                key = param_name if param_name in kwargs else par

                value = kwargs.get(key, default_value)
                if value is None:
                    raise ValueError(f"Missing initial value for parameter '{key}'")

                vary = kwargs.get(f"{key}_vary", True)
                use_log = kwargs.get(f"{key}_log", par == "mass")
                vmin = kwargs.get(f"{key}_min", -np.inf)
                vmax = kwargs.get(f"{key}_max", np.inf)

                if use_log:
                    if value <= 0:
                        raise ValueError(f"Parameter '{key}' must be > 0 when using log-space fit")
                    if np.isfinite(vmin) and vmin <= 0:
                        raise ValueError(f"Lower bound '{key}_min' must be > 0 when using log-space fit")
                    if np.isfinite(vmax) and vmax <= 0:
                        raise ValueError(f"Upper bound '{key}_max' must be > 0 when using log-space fit")
                    value = np.log10(value)
                    if np.isfinite(vmin):
                        vmin = np.log10(vmin)
                    if np.isfinite(vmax):
                        vmax = np.log10(vmax)

                params.add(param_name, value=value, vary=vary, min=vmin, max=vmax)
                param_refs.append((model, par, param_name, use_log))

        def cost_function(p):
            for model, par, param_name, use_log in param_refs:
                par_val = p[param_name].value
                if use_log:
                    par_val = 10 ** par_val
                setattr(model, par, par_val)

            model_density = np.asarray(self.density(r))
            input_density = np.asarray(input_model.density(r))
            valid = (model_density > 0) & (input_density > 0)
            if not np.any(valid):
                raise ValueError("Density models do not overlap in positive domain for log-space fit")
            return (np.log10(model_density[valid]) - np.log10(input_density[valid])) ** 2

        mini = lmfit.Minimizer(cost_function, params)
        result = mini.minimize()

        # ensure best-fit parameters are written back even if minimizer exits early
        for model, par, param_name, use_log in param_refs:
            par_val = result.params[param_name].value
            if use_log:
                par_val = 10 ** par_val
            setattr(model, par, par_val)

        return result
