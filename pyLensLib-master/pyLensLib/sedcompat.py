"""
Lightweight replacement for the subset of ``barak.sed`` used by pyLensLib.

Implemented features:
- ``Passband`` with wavelength/transmission loading and ``effective_wa``
- ``SED`` with ``copy``, ``redshift_to``, ``calc_flux``, ``calc_mag``,
  and ``normalise_to_mag``
- helper functions ``mag2flux``, ``flambda_to_fnu``, ``fnu_to_flambda``
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Tuple, Union
import os

import numpy as np


# cgs constants
_C_CGS = 2.99792458e10  # cm / s
_JY_CGS = 1e-23         # erg / s / cm^2 / Hz
_AB_FNU = 3631.0 * _JY_CGS


ArrayLike = Union[np.ndarray, list, tuple]


def _as_sorted_arrays(wa: ArrayLike, tr: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    wa_ = np.asarray(wa, dtype=float)
    tr_ = np.asarray(tr, dtype=float)
    if wa_.ndim != 1 or tr_.ndim != 1:
        raise ValueError("wavelength and transmission must be 1D arrays")
    if wa_.size != tr_.size:
        raise ValueError("wavelength and transmission arrays must have the same size")
    isort = np.argsort(wa_)
    wa_ = wa_[isort]
    tr_ = tr_[isort]
    return wa_, tr_


def _resolve_passband_path(source: Union[str, Path]) -> Path:
    """
    Resolve passband path with BPZ-style fallbacks.
    """
    src = Path(str(source)).expanduser()
    if src.is_file():
        return src

    rel = str(source)
    if rel.startswith("BPZ/"):
        rel = rel.split("/", 1)[1]

    env_dirs = [
        os.environ.get("PYLENSLIB_FILTER_DIR"),
        os.environ.get("BPZ_FILTER_DIR"),
        os.environ.get("BPZDIR"),
        os.environ.get("BPZ_PATH"),
    ]
    candidates = [Path.cwd() / rel, Path.cwd() / Path(rel).name]
    for d in env_dirs:
        if not d:
            continue
        base = Path(d).expanduser()
        candidates.extend(
            [
                base / rel,
                base / Path(rel).name,
                base / "FILTER" / rel,
                base / "FILTER" / Path(rel).name,
            ]
        )

    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        f"Passband file '{source}' not found. "
        "Set an absolute path, or configure BPZDIR/BPZ_FILTER_DIR/PYLENSLIB_FILTER_DIR."
    )


def fnu_to_flambda(wa: ArrayLike, f_nu: ArrayLike) -> np.ndarray:
    wa_ = np.asarray(wa, dtype=float)
    f_nu_ = np.asarray(f_nu, dtype=float)
    return _C_CGS / (wa_ * 1e-8) ** 2 * f_nu_ * 1e-8


def flambda_to_fnu(wa: ArrayLike, f_lambda: ArrayLike) -> np.ndarray:
    wa_ = np.asarray(wa, dtype=float)
    f_lambda_ = np.asarray(f_lambda, dtype=float)
    return (wa_ * 1e-8) ** 2 * f_lambda_ * 1e8 / _C_CGS


def effective_wa(wa: ArrayLike, tr: ArrayLike) -> float:
    wa_, tr_ = _as_sorted_arrays(wa, tr)
    den = np.trapz(tr_ / wa_, wa_)
    num = np.trapz(tr_ * wa_, wa_)
    if den <= 0.0 or num <= 0.0:
        return np.nan
    return float(np.sqrt(num / den))


def _ab_reference_flux(wa: np.ndarray, tr: np.ndarray) -> float:
    """
    AB reference flux in the passband for the flux estimator:
      F = ∫ T(λ) f_λ(λ) λ dλ / ∫ T(λ) λ dλ
    """
    i1 = np.trapz(tr / wa, wa)
    i2 = np.trapz(tr * wa, wa)
    if i1 <= 0.0 or i2 <= 0.0:
        return np.nan
    return float(_C_CGS * _AB_FNU * 1e8 * i1 / i2)


def _calc_flux_single(wa_pass: np.ndarray, tr_pass: np.ndarray, wa_sed: np.ndarray, fl_sed: np.ndarray) -> float:
    wa_sed = np.asarray(wa_sed, dtype=float)
    fl_sed = np.asarray(fl_sed, dtype=float)
    if wa_sed.ndim != 1 or fl_sed.ndim != 1:
        raise ValueError("SED wavelength and flux must be 1D arrays")
    if wa_sed.size != fl_sed.size:
        raise ValueError("SED wavelength and flux arrays must have same size")

    isort = np.argsort(wa_sed)
    wa_sed = wa_sed[isort]
    fl_sed = fl_sed[isort]

    wmin = wa_pass[0]
    wmax = wa_pass[-1]
    if wa_sed[0] > wmin or wa_sed[-1] < wmax:
        warnings.warn("SED does not cover the whole bandpass, extrapolating", RuntimeWarning)
        dw = np.median(np.diff(wa_sed))
        if not np.isfinite(dw) or dw <= 0:
            dw = np.median(np.diff(wa_pass))
        wa_eval = np.arange(wmin, wmax + dw, dw)
        fl_eval = np.interp(wa_eval, wa_sed, fl_sed)
    else:
        i, j = wa_sed.searchsorted([wmin, wmax])
        if j - i < 2:
            wa_eval = wa_pass
            fl_eval = np.interp(wa_eval, wa_sed, fl_sed)
        else:
            wa_eval = wa_sed[i:j]
            fl_eval = fl_sed[i:j]

    dw_band = np.median(np.diff(wa_pass))
    dw_sed = np.median(np.diff(wa_eval))
    if np.isfinite(dw_sed) and np.isfinite(dw_band) and dw_sed > dw_band and dw_band > 20:
        warnings.warn(
            f"SED wavelength sampling interval ~{dw_sed:.2f} Ang, "
            f"but bandpass sampling interval ~{dw_band:.2f} Ang",
            RuntimeWarning,
        )
        wa_grid = wa_pass
        fl_grid = np.interp(wa_grid, wa_eval, fl_eval)
        tr_grid = tr_pass
    else:
        wa_grid = wa_eval
        fl_grid = fl_eval
        tr_grid = np.interp(wa_grid, wa_pass, tr_pass)

    denom = np.trapz(tr_grid * wa_grid, wa_grid)
    if denom <= 0.0:
        return np.nan
    return float(np.trapz(tr_grid * fl_grid * wa_grid, wa_grid) / denom)


def calc_flux(passband: "Passband", wa: ArrayLike, fl: ArrayLike) -> float:
    return _calc_flux_single(passband.wa, passband.tr, np.asarray(wa, dtype=float), np.asarray(fl, dtype=float))


def mag2flux(abmag: Union[float, np.ndarray], passband: "Passband") -> np.ndarray:
    """
    Convert AB magnitude to f_lambda at the passband effective wavelength.
    """
    abmag_ = np.asarray(abmag, dtype=float)
    fnu = 10.0 ** (-(abmag_ + 48.6) / 2.5)
    return fnu_to_flambda(passband.effective_wa, fnu)


@dataclass
class Passband:
    source: Union[str, Path, None] = None
    wa: np.ndarray | None = None
    tr: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.wa is None or self.tr is None:
            if self.source is None:
                raise ValueError("Passband requires either source path or wa/tr arrays")
            path = _resolve_passband_path(self.source)
            data = np.loadtxt(str(path), usecols=(0, 1), unpack=True)
            wa_, tr_ = data
        else:
            wa_, tr_ = self.wa, self.tr
        wa_, tr_ = _as_sorted_arrays(wa_, tr_)
        self.wa = wa_
        self.tr = tr_
        self.effective_wa = effective_wa(self.wa, self.tr)
        self._ab_flux = _ab_reference_flux(self.wa, self.tr)


@dataclass
class SED:
    wa: np.ndarray
    fl: np.ndarray
    z: float = 0.0

    def __post_init__(self) -> None:
        self.wa = np.asarray(self.wa, dtype=float).copy()
        self.fl = np.asarray(self.fl, dtype=float).copy()
        if self.wa.ndim != 1 or self.fl.ndim != 1:
            raise ValueError("SED wa/fl must be 1D arrays")
        if self.wa.size != self.fl.size:
            raise ValueError("SED wa/fl must have the same size")
        isort = np.argsort(self.wa)
        self.wa = self.wa[isort]
        self.fl = self.fl[isort]
        self.z = float(self.z)

    def copy(self) -> "SED":
        return SED(self.wa.copy(), self.fl.copy(), z=self.z)

    def redshift_to(self, newz: float) -> None:
        newz = float(newz)
        if newz < -0.999999:
            raise ValueError("new redshift must be > -1")
        scale = (1.0 + newz) / (1.0 + self.z)
        self.wa *= scale
        self.fl /= scale
        self.z = newz

    def calc_flux(self, passband: Passband) -> float:
        return calc_flux(passband, self.wa, self.fl)

    def calc_mag(self, passband: Passband, system: str = "AB") -> float:
        if system.upper() != "AB":
            raise NotImplementedError("Only AB magnitudes are supported")
        f1 = self.calc_flux(passband)
        if not np.isfinite(f1) or f1 <= 0.0 or not np.isfinite(passband._ab_flux) or passband._ab_flux <= 0.0:
            return np.inf
        return float(-2.5 * np.log10(f1 / passband._ab_flux))

    def normalise_to_mag(self, mag: float, passband: Passband, system: str = "AB") -> None:
        if system.upper() != "AB":
            raise NotImplementedError("Only AB magnitudes are supported")
        f_now = self.calc_flux(passband)
        if not np.isfinite(f_now) or f_now <= 0.0:
            raise ValueError("Current SED flux in passband is non-positive; cannot normalize")
        f_target = passband._ab_flux * 10.0 ** (-0.4 * float(mag))
        self.fl *= (f_target / f_now)
