"""Gravitational-wave source model for pyLensLib."""

import numpy as np
from scipy.ndimage import map_coordinates

from pyLensLib.gensrc import gensrc
from pyLensLib.pointsrc import pointsrc
from pyLensLib.gwlensing import (
    geometric_optics_amplification,
    morse_indices_from_lens_fields,
    newtonian_fd_waveform,
    point_mass_einstein_radius_arcsec,
    point_mass_geometric_amplification,
    point_mass_regime,
    point_mass_wave_amplification,
    weak_lensing_amplification,
)


class gwsource(gensrc):
    """Frequency-domain gravitational-wave source with lensing support.

    The class stores an unlensed strain ``h(f)`` and can apply weak lensing,
    geometric-optics strong lensing, point-mass wave optics, or a simple
    macro-plus-wave hybrid amplification.

    Parameters
    ----------
    ys1, ys2 : float
        Source-plane angular position in the same units used by the deflector
        grid, normally arcsec.
    zs : float
        Source redshift.
    gl : object, optional
        pyLensLib lens/deflector object.  Geometric-optics lensing reuses the
        existing point-source image finder and the lens time-delay surface.
    waveform : callable, tuple, dict, or array-like, optional
        User-supplied frequency-domain waveform.  A callable must accept the
        frequency array and return complex strain.  A tuple should be
        ``(frequencies, strain)``.  A dict may contain ``frequencies``/``f`` and
        ``strain``/``h``.
    wave_lens : dict, optional
        Compact-lens parameters for wave optics.  The first implementation
        supports ``{"model": "point_mass", "mass": ..., "y": ...}``.
    lensing, regime : str, optional
        ``lensing`` can force ``none``, ``weak``, ``geometric``, ``wave``, or
        ``hybrid``.  ``regime="auto"`` chooses among those modes.
    """

    _REGIME_ALIASES = {
        None: "none",
        False: "none",
        True: "auto",
        "none": "none",
        "unlensed": "none",
        "auto": "auto",
        "weak": "weak",
        "strong": "geometric",
        "optic": "geometric",
        "optical": "geometric",
        "geometric": "geometric",
        "geo": "geometric",
        "wave": "wave",
        "wave_optics": "wave",
        "hybrid": "hybrid",
    }

    def __init__(
        self,
        ys1=0.0,
        ys2=0.0,
        zs=1.0,
        gl=None,
        m1=30.0,
        m2=30.0,
        luminosity_distance=None,
        waveform=None,
        waveform_model="newtonian",
        frequencies=None,
        h_unlensed=None,
        fmin=10.0,
        fmax=1024.0,
        delta_f=1.0 / 8.0,
        lensing="auto",
        regime="auto",
        wave_lens=None,
        use_lenstronomy=False,
        refine_images=True,
        refine_to_td=True,
        wave_threshold=10.0,
        geo_threshold=100.0,
        **kwargs,
    ):
        super().__init__()
        self.ys1 = float(ys1)
        self.ys2 = float(ys2)
        self.zs = float(zs)
        self.df = gl
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.luminosity_distance = luminosity_distance
        self.waveform = waveform
        self.waveform_model = waveform_model
        self.h_unlensed_input = h_unlensed
        self.lensing = self._normalize_regime(lensing)
        self.regime = self._normalize_regime(regime)
        self.wave_lens = dict(wave_lens) if wave_lens is not None else None
        self.use_lenstronomy = use_lenstronomy
        self.refine_images = refine_images
        self.refine_to_td = refine_to_td
        self.wave_threshold = float(wave_threshold)
        self.geo_threshold = float(geo_threshold)
        self.waveform_kwargs = kwargs

        self.rescf = 1.0
        if gl is not None and hasattr(gl, "co") and self.zs != getattr(gl, "zs", self.zs):
            if self.zs > gl.zl:
                ds = gl.co.angular_diameter_distance(self.zs).value
                dls = gl.co.angular_diameter_distance_z1z2(gl.zl, self.zs).value
                self.rescf = dls / ds * gl.ds.value / gl.dls.value
            else:
                self.rescf = 0.0

        waveform_frequencies = self._frequencies_from_waveform(waveform)
        if frequencies is None and waveform_frequencies is not None:
            frequencies = waveform_frequencies
        self.frequencies = self._make_frequency_grid(frequencies, fmin, fmax, delta_f)
        self.frequency = self.frequencies

        self._images_cache = None
        self._pointsrc_cache = None

        self.strain_unlensed = self._build_unlensed_waveform()
        self.h_unlensed = self.strain_unlensed
        self.strain_lensed = None
        self.h_lensed = None
        self.amplification = None
        self.regime_selected = self.select_lensing_regime()

    def _normalize_regime(self, value):
        if isinstance(value, str):
            key = value.lower()
        else:
            key = value
        if key not in self._REGIME_ALIASES:
            raise ValueError(
                "Unsupported GW lensing regime '{}'. Use one of none, auto, "
                "weak, geometric, wave, or hybrid.".format(value)
            )
        return self._REGIME_ALIASES[key]

    def _make_frequency_grid(self, frequencies, fmin, fmax, delta_f):
        if frequencies is not None:
            freq = np.asarray(frequencies, dtype=float)
        else:
            freq = np.arange(float(fmin), float(fmax) + 0.5 * float(delta_f), float(delta_f))
        if freq.ndim != 1 or freq.size == 0:
            raise ValueError("frequencies must be a non-empty one-dimensional array")
        return freq

    def _frequencies_from_waveform(self, waveform):
        if waveform is None or callable(waveform):
            return None
        if isinstance(waveform, tuple) and len(waveform) == 2:
            return np.asarray(waveform[0], dtype=float)
        if isinstance(waveform, dict):
            for key in ("frequencies", "frequency", "f"):
                if key in waveform:
                    return np.asarray(waveform[key], dtype=float)
        return None

    def _build_unlensed_waveform(self):
        if self.h_unlensed_input is not None:
            strain = np.asarray(self.h_unlensed_input, dtype=np.complex128)
        elif self.waveform is None:
            if self.waveform_model != "newtonian":
                raise ValueError(
                    "Only waveform_model='newtonian' is built in. Pass a "
                    "frequency-domain waveform for external models."
                )
            strain = newtonian_fd_waveform(
                self.frequencies,
                m1=self.m1,
                m2=self.m2,
                zs=self.zs,
                luminosity_distance=self.luminosity_distance,
                **self.waveform_kwargs,
            )
        elif callable(self.waveform):
            strain = np.asarray(self.waveform(self.frequencies), dtype=np.complex128)
        elif isinstance(self.waveform, tuple) and len(self.waveform) == 2:
            freq, strain = self.waveform
            freq = np.asarray(freq, dtype=float)
            strain = np.asarray(strain, dtype=np.complex128)
            if not np.allclose(freq, self.frequencies):
                raise ValueError("waveform frequencies do not match the gwsource frequency grid")
        elif isinstance(self.waveform, dict):
            for key in ("strain", "h", "waveform"):
                if key in self.waveform:
                    strain = np.asarray(self.waveform[key], dtype=np.complex128)
                    break
            else:
                raise ValueError("waveform dict must contain strain, h, or waveform")
        else:
            strain = np.asarray(self.waveform, dtype=np.complex128)

        if strain.shape != self.frequencies.shape:
            raise ValueError("waveform strain must have the same shape as the frequency grid")
        return strain

    def unlensed_waveform(self):
        """Return ``(frequencies, h_unlensed)``."""
        return self.frequencies.copy(), self.strain_unlensed.copy()

    def simulate_unlensed_waveform(self):
        """Alias for :meth:`unlensed_waveform`."""
        return self.unlensed_waveform()

    def lensed_waveform(self, regime=None):
        """Return ``(frequencies, h_lensed)`` for the chosen lensing regime."""
        amplification = self.amplification_factor(self.frequencies, regime=regime)
        self.amplification = amplification
        self.strain_lensed = amplification * self.strain_unlensed
        self.h_lensed = self.strain_lensed
        return self.frequencies.copy(), self.strain_lensed.copy()

    def simulate_lensed_waveform(self, regime=None):
        """Alias for :meth:`lensed_waveform`."""
        return self.lensed_waveform(regime=regime)

    def select_lensing_regime(self, frequencies=None):
        """Choose the lensing regime for the current source configuration."""
        frequencies = self.frequencies if frequencies is None else np.asarray(frequencies, dtype=float)
        if self.lensing == "none" or (self.df is None and self.wave_lens is None):
            return "none"
        if self.regime != "auto":
            return self.regime
        if self.lensing != "auto":
            return self.lensing

        if self.wave_lens is not None:
            mass, zl, _ = self._point_mass_lens_parameters(require_y=False)
            compact_regime = point_mass_regime(
                frequencies,
                mass,
                zl=zl,
                wave_threshold=self.wave_threshold,
                geo_threshold=self.geo_threshold,
            )
            if self.df is not None and compact_regime in ("wave", "hybrid"):
                return "hybrid"
            return compact_regime

        if self.df is not None:
            try:
                images = self.find_geometric_images()
                if images["theta1"].size > 1:
                    return "geometric"
            except Exception:
                pass
            return "weak"

        return "none"

    def amplification_factor(self, frequencies=None, regime=None):
        """Return the complex GW lensing amplification ``F(f)``."""
        frequencies = self.frequencies if frequencies is None else np.asarray(frequencies, dtype=float)
        selected = self._normalize_regime(regime) if regime is not None else self.select_lensing_regime(frequencies)
        self.regime_selected = selected

        if selected == "none":
            return np.ones_like(frequencies, dtype=np.complex128)

        if selected == "weak":
            mu = self.weak_magnification()
            return np.ones_like(frequencies, dtype=np.complex128) * weak_lensing_amplification(mu)

        if selected == "wave":
            return self.wave_amplification(frequencies)

        if selected == "geometric":
            return self.geometric_amplification(frequencies)

        if selected == "hybrid":
            wave_amp = self.wave_amplification(frequencies)
            if self.df is None:
                return wave_amp
            try:
                macro_amp = self.geometric_amplification(frequencies)
            except Exception:
                macro_amp = np.ones_like(frequencies, dtype=np.complex128) * weak_lensing_amplification(
                    self.weak_magnification()
                )
            return macro_amp * wave_amp

        raise ValueError("Unsupported selected regime '{}'".format(selected))

    def wave_amplification(self, frequencies=None):
        """Return a point-mass wave-optics amplification factor."""
        if self.wave_lens is None:
            raise RuntimeError("wave_lens parameters are required for wave-optics GW lensing")
        frequencies = self.frequencies if frequencies is None else np.asarray(frequencies, dtype=float)
        mass, zl, y = self._point_mass_lens_parameters(require_y=True)
        return point_mass_wave_amplification(frequencies, mass, y, zl=zl)

    def geometric_amplification(self, frequencies=None):
        """Return the geometric-optics amplification factor."""
        frequencies = self.frequencies if frequencies is None else np.asarray(frequencies, dtype=float)

        if self.df is None:
            if self.wave_lens is None:
                raise RuntimeError("A deflector or point-mass wave_lens is required for geometric lensing")
            mass, zl, y = self._point_mass_lens_parameters(require_y=True)
            return point_mass_geometric_amplification(frequencies, mass, y, zl=zl)

        images = self.find_geometric_images()
        if images["theta1"].size == 0:
            raise RuntimeError("No geometric-optics images were found for this source")
        delays = self.image_time_delays(images=images, relative=True, units="s")
        return geometric_optics_amplification(
            frequencies,
            images["magnification"],
            delays,
            morse_indices=images["morse_index"],
            normalize_delays=False,
        )

    def find_geometric_images(self, refresh=False):
        """Find geometric-optics images using the existing point-source solver."""
        if self.df is None:
            raise RuntimeError("A pyLensLib deflector is required to find geometric images")
        if self._images_cache is not None and not refresh:
            return self._copy_images(self._images_cache)

        if hasattr(self.df, "nray1") and hasattr(self.df, "nray2"):
            nx = int(self.df.nray1)
            ny = int(self.df.nray2)
        else:
            nx = int(self.df.a1.shape[1])
            ny = int(self.df.a1.shape[0])
        size = float(max(getattr(self.df, "size1", nx), getattr(self.df, "size2", ny)))
        use_td = bool(self.refine_to_td and getattr(self.df, "computed_potential", False))

        ps = pointsrc(
            size=size,
            Npix=max(nx, ny),
            gl=self.df,
            ys1=self.ys1,
            ys2=self.ys2,
            zs=self.zs,
            flux=1.0,
            use_lenstronomy=self.use_lenstronomy,
            refine=self.refine_images,
            refine_to_td=use_td,
        )

        theta1 = np.asarray(ps.xi1, dtype=float)
        theta2 = np.asarray(ps.xi2, dtype=float)
        magnification = self.signed_magnification(theta1, theta2)
        kappa, gamma1, gamma2 = self.lens_fields(theta1, theta2)
        morse_index = morse_indices_from_lens_fields(kappa, gamma1, gamma2)

        images = {
            "theta1": theta1,
            "theta2": theta2,
            "magnification": np.asarray(magnification, dtype=float),
            "morse_index": np.asarray(morse_index, dtype=int),
        }
        self._images_cache = self._copy_images(images)
        self._pointsrc_cache = ps
        return self._copy_images(images)

    def lens_fields(self, theta1, theta2):
        """Interpolate ``kappa``, ``gamma1``, and ``gamma2`` at angular positions."""
        if self.df is None:
            raise RuntimeError("A deflector is required to interpolate lens fields")
        theta1 = np.atleast_1d(np.asarray(theta1, dtype=float))
        theta2 = np.atleast_1d(np.asarray(theta2, dtype=float))
        xpix = (theta1 - self.df.thetax[0]) / self.df.pixel_scale
        ypix = (theta2 - self.df.thetay[0]) / self.df.pixel_scale
        coords = [ypix, xpix]
        kappa = map_coordinates(self.df.ka, coords, order=1, mode="nearest", prefilter=True)
        gamma1 = map_coordinates(self.df.g1, coords, order=1, mode="nearest", prefilter=True)
        gamma2 = map_coordinates(self.df.g2, coords, order=1, mode="nearest", prefilter=True)
        return kappa, gamma1, gamma2

    def signed_magnification(self, theta1, theta2):
        """Return signed lensing magnification at angular positions."""
        kappa, gamma1, gamma2 = self.lens_fields(theta1, theta2)
        det_a = (1.0 - kappa) ** 2 - gamma1**2 - gamma2**2
        return 1.0 / det_a

    def weak_magnification(self, theta1=None, theta2=None):
        """Return a scalar weak-lensing magnification.

        If no position is supplied, the lens fields are evaluated at the source
        angular position as a weak-lensing approximation.
        """
        if self.df is None:
            return 1.0
        if theta1 is None:
            theta1 = self.ys1
        if theta2 is None:
            theta2 = self.ys2
        return float(np.asarray(self.signed_magnification(theta1, theta2)).ravel()[0])

    def image_time_delays(self, images=None, relative=True, units="s"):
        """Return image time delays from the deflector time-delay surface."""
        if self.df is None:
            raise RuntimeError("A deflector is required to compute image time delays")
        if not getattr(self.df, "computed_potential", False):
            raise RuntimeError(
                "Geometric GW lensing requires a deflector with a potential. "
                "Build the lens with compute_potential=True or provide a potential map."
            )
        if images is None:
            images = self.find_geometric_images()
        td_surface = self.df.t_delay_surf(beta=(self.ys1, self.ys2))
        theta1 = np.asarray(images["theta1"], dtype=float)
        theta2 = np.asarray(images["theta2"], dtype=float)
        xpix = (theta1 - self.df.thetax[0]) / self.df.pixel_scale
        ypix = (theta2 - self.df.thetay[0]) / self.df.pixel_scale
        delays = map_coordinates(td_surface, [ypix, xpix], order=1, mode="nearest", prefilter=True)
        if relative and delays.size > 0:
            delays = delays - np.nanmin(delays)
        if units in ("s", "sec", "second", "seconds"):
            return delays * 86400.0
        if units in ("d", "day", "days"):
            return delays
        raise ValueError("units must be seconds ('s') or days ('d')")

    def _point_mass_lens_parameters(self, require_y=True):
        if self.wave_lens is None:
            raise RuntimeError("wave_lens parameters are required")
        model = self.wave_lens.get("model", self.wave_lens.get("profile", "point_mass"))
        if str(model).lower() not in ("point_mass", "pointmass", "pm"):
            raise ValueError("Only point-mass wave_lens models are currently supported")
        for key in ("mass", "mass_msun", "M", "M_msun"):
            if key in self.wave_lens:
                mass = float(self.wave_lens[key])
                break
        else:
            raise ValueError("wave_lens must include a point-lens mass in solar masses")
        zl = float(self.wave_lens.get("zl", getattr(self.df, "zl", 0.0)))
        y = self._point_mass_y(mass, zl) if require_y else self.wave_lens.get("y", None)
        return mass, zl, y

    def _point_mass_y(self, mass, zl):
        if "y" in self.wave_lens:
            return float(self.wave_lens["y"])
        for key in ("theta_e_arcsec", "theta_E_arcsec", "theta_e", "theta_E"):
            if key in self.wave_lens:
                theta_e = float(self.wave_lens[key])
                break
        else:
            if self.df is None or not hasattr(self.df, "co"):
                raise ValueError(
                    "wave_lens must include y or theta_e_arcsec when no cosmology-bearing deflector is available"
                )
            theta_e = point_mass_einstein_radius_arcsec(mass, zl, self.zs, self.df.co)

        beta = self.wave_lens.get("beta_arcsec", None)
        if beta is None:
            lens_theta1 = float(self.wave_lens.get("theta1", self.wave_lens.get("x", 0.0)))
            lens_theta2 = float(self.wave_lens.get("theta2", self.wave_lens.get("y_center", 0.0)))
            beta = np.hypot(self.ys1 - lens_theta1, self.ys2 - lens_theta2)
        theta_e = float(theta_e)
        if theta_e <= 0.0:
            raise ValueError("point-mass Einstein radius must be positive")
        return float(beta) / theta_e

    def _copy_images(self, images):
        return {key: np.array(value, copy=True) for key, value in images.items()}


GWSource = gwsource
