#!/usr/bin/env python3
"""
clumPyLen workflow as a pyLensLib app.

This script keeps the clumpy-galaxy workflow that lived in the standalone
clumPyLen repo, but uses pyLensLib as the canonical home for the reusable
science objects. The hot paths are tuned to avoid repeated file rewrites,
per-clump SED reloads, and unnecessary point-source work on runs that do not
need it.

Bundled example inputs live in apps/clumPyLens/input. The YAML input files are
organized into sections, and each parameter can carry a human-readable
description alongside its value.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from math import lgamma
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import Distance, SkyCoord
from astropy.io import fits
from scipy.special import gammaln
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy.signal import fftconvolve
from tqdm import tqdm
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pyLensLib.lenstool import create_deflector
from pyLensLib.observation import observation
from pyLensLib.pointsrc import pointsrc
from pyLensLib.samplers import sample2Dimage
from pyLensLib.spiral import spiral_modifier_from_grid

try:  # pragma: no cover - optional acceleration
    from pyLensLib.sersic_numba import sersic as SersicModel
except Exception:  # pragma: no cover - fallback when numba is unavailable
    from pyLensLib.sersic import sersic as SersicModel

try:  # pragma: no cover - optional dependency
    import noise as _noise
except Exception:  # pragma: no cover - fallback path
    _noise = None

try:  # pragma: no cover - optional dependency
    from photutils.psf.matching import resize_psf
except Exception:  # pragma: no cover - fallback path

    def resize_psf(psf, in_pix_scale, out_pix_scale, order=1):
        scale = in_pix_scale / out_pix_scale
        new_shape = tuple(max(1, int(round(dim * scale))) for dim in psf.shape)
        resized = zoom(psf, zoom=scale, order=order)
        if resized.shape != new_shape:
            resized = resize_psf._trim_or_pad(resized, new_shape)
        total = resized.sum()
        if total > 0:
            resized = resized / total
        return resized

    def _trim_or_pad(arr, target_shape):
        out = np.zeros(target_shape, dtype=arr.dtype)
        ny = min(target_shape[0], arr.shape[0])
        nx = min(target_shape[1], arr.shape[1])
        y0 = (target_shape[0] - ny) // 2
        x0 = (target_shape[1] - nx) // 2
        ay0 = (arr.shape[0] - ny) // 2
        ax0 = (arr.shape[1] - nx) // 2
        out[y0 : y0 + ny, x0 : x0 + nx] = arr[ay0 : ay0 + ny, ax0 : ax0 + nx]
        return out

    resize_psf._trim_or_pad = _trim_or_pad  # type: ignore[attr-defined]

try:  # pragma: no cover - optional acceleration
    from numba import njit
except Exception:  # pragma: no cover - fallback path
    njit = None


TELESCOPE_NAMES = ["HST_ACS", "HST_WFC3", "JWST_SW", "JWST_LW"]
FILTERS_BY_TELESCOPE = {
    "JWST_SW": ["F090W", "F150W", "F200W", "F115W"],
    "JWST_LW": ["F277W", "F356W", "F444W", "F410M"],
    "HST_WFC3": ["F105W", "F125W", "F160W"],
    "HST_ACS": ["F435W", "F606W", "F814W"],
}
YGG_MAG_COLUMNS = {
    "JWST_SW": [2, 3, 4, 5],
    "JWST_LW": [2, 3, 4, 5],
    "HST_WFC3": [2, 3, 4],
    "HST_ACS": [2, 3, 4],
}

DEFAULT_EXAMPLE_INPUT_FILE = Path(__file__).resolve().parent / "input" / "input_file.yaml"

YAML_METADATA_KEYS = {"description", "desc", "note", "notes", "help", "comment"}
MIN_HOST_RENDER_SCALE = 0.25

EXPECTED_INPUT_KEYS = {
    "PATH_TO_def_angle",
    "PATH_TO_input_file",
    "PATH_TO_sed",
    "PATH_TO_psf",
    "PATH_TO_clumps",
    "PATH_TO_fits",
    "PATH_TO_host",
    "PATH_TO_images",
    "PATH_TO_needed",
    "ask_Perlin",
    "perlin_scale",
    "perlin_octaves",
    "perlin_persistence",
    "perlin_lacunarity",
    "perlin_seed",
    "ask_spiral",
    "Na",
    "spiral_A",
    "spiral_alpha",
    "spiral_Phid",
    "pix_scale",
    "pix_scale_unl",
    "cluster",
    "zl",
    "coords",
    "blue_",
    "green_",
    "red_",
    "mass_total",
    "z_s",
    "sizex",
    "sizey",
    "size_unl",
    "gal_x",
    "gal_y",
    "Mass_bul",
    "Mass_bul_Y",
    "Re_bul",
    "n_bul",
    "q_bul",
    "pa_bul",
    "Mag_Y_1Gy_J_SW",
    "Mag_Y_1Gy_J_LW",
    "Mag_Y_1Gy_H_ACS",
    "Mag_Y_1Gy_H_WFC3",
    "Mass_disk",
    "Mass_disk_Y",
    "Re_disk",
    "n_disk",
    "q_disk",
    "disk_inclination_deg",
    "pa_disk",
    "mass_min",
    "mass_max",
    "trunc_mass",
    "alpha",
    "gamma",
    "delta",
    "alpha_t",
    "t_min",
    "t_max",
    "Mag_Y_05Gy_J_SW",
    "Mag_Y_05Gy_J_LW",
    "Mag_Y_05Gy_H_ACS",
    "Mag_Y_05Gy_H_WFC3",
    "fsub",
    "Texp_H_ACS",
    "Flux_sky_H_ACS",
    "Texp_H_WFC3",
    "Flux_sky_H_WFC3",
    "Texp_J_SW",
    "Texp_J_LW",
    "Flux_sky_J_SW",
    "Flux_sky_J_LW",
    "ZP_J_SW",
    "ZP_J_LW",
    "ZP_H_ACS",
    "ZP_H_WFC3",
    "pix_scl_instr",
    "pix_scl_PSF_J_SW",
    "pix_scl_PSF_J_LW",
    "pix_scl_PSF_H_ACS",
    "pix_scl_PSF_H_WFC3",
}


@dataclass(frozen=True)
class RenderGridCache:
    lensed_y1: np.ndarray
    lensed_y2: np.ndarray
    unlensed_y1: np.ndarray
    unlensed_y2: np.ndarray
    lensed_shape: Tuple[int, int]
    unlensed_shape: Tuple[int, int]
    pix_scale: float
    pix_scale_unl: float


@dataclass(frozen=True)
class SersicBatchParams:
    ys1: np.ndarray
    ys2: np.ndarray
    re: np.ndarray
    n: np.ndarray
    q: np.ndarray
    pa: np.ndarray
    flux: np.ndarray
    amp: np.ndarray
    bn: np.ndarray
    inv_re: np.ndarray
    inv_n: np.ndarray
    cos_pa: np.ndarray
    sin_pa: np.ndarray
    inv_q: np.ndarray
    rmax: float


def _sersic_rfun(c: np.ndarray | float) -> np.ndarray | float:
    c_arr = np.asarray(c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.pi * (c_arr + 2.0) / 4.0 / np.exp(
            gammaln(1.0 / (c_arr + 2.0)) + gammaln(1.0 + 1.0 / (c_arr + 2.0)) - gammaln(1.0 + 2.0 / (c_arr + 2.0))
        )
    return out if np.ndim(c_arr) else float(out)


def _build_sersic_amplitude(flux: np.ndarray, re: np.ndarray, n: np.ndarray, q: np.ndarray, c: np.ndarray, pix_scale: float) -> np.ndarray:
    bn = 1.992 * n - 0.3271
    rfun = _sersic_rfun(c)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        sie = flux / (
            2.0
            * np.pi
            * re**2
            * np.exp(bn)
            * n
            * bn ** (-2.0 * n)
            * np.exp(gammaln(2.0 * n))
            * q
        )
    return np.asarray(sie * rfun * pix_scale * pix_scale, dtype=float)


def _build_batch_params(
    *,
    catalog: Dict[str, Any],
    flux: np.ndarray,
    pix_scale: float,
    rmaxf: float,
) -> SersicBatchParams:
    re = np.asarray(catalog["reff_arcsec"], dtype=float)
    n = np.asarray(catalog["sersic_index"], dtype=float)
    q = np.asarray(catalog["axis_ratio"], dtype=float)
    pa = np.asarray(catalog["position_angle"], dtype=float)
    ys1 = np.asarray(catalog["pos_x_arcsec"], dtype=float)
    ys2 = np.asarray(catalog["pos_y_arcsec"], dtype=float)
    c = np.zeros_like(re)
    inv_re = np.divide(1.0, re, out=np.zeros_like(re), where=re != 0.0)
    inv_n = np.divide(1.0, n, out=np.zeros_like(n), where=n != 0.0)
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    inv_q = np.divide(1.0, q, out=np.zeros_like(q), where=q != 0.0)
    amp = _build_sersic_amplitude(np.asarray(flux, dtype=float), re, n, q, c, pix_scale)
    bn = 1.992 * n - 0.3271
    return SersicBatchParams(
        ys1=ys1,
        ys2=ys2,
        re=re,
        n=n,
        q=q,
        pa=pa,
        flux=np.asarray(flux, dtype=float),
        amp=amp,
        bn=bn,
        inv_re=inv_re,
        inv_n=inv_n,
        cos_pa=cos_pa,
        sin_pa=sin_pa,
        inv_q=inv_q,
        rmax=float(rmaxf),
    )


def _extract_parameter_key(description: str) -> Optional[str]:
    tokens = description.strip().split()
    if not tokens:
        return None
    for token in reversed(tokens):
        if token in EXPECTED_INPUT_KEYS or token.startswith("PATH_TO_"):
            return token
    for token in reversed(tokens):
        cleaned = token.strip(",:;()[]{}")
        if cleaned in EXPECTED_INPUT_KEYS or cleaned.startswith("PATH_TO_"):
            return cleaned
    return tokens[-1].strip(",:;()[]{}")


def _collect_yaml_parameters(node: Any, parameters: Dict[str, Any]) -> None:
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key in YAML_METADATA_KEYS:
            continue
        if isinstance(value, dict):
            if "value" in value:
                parameters[key] = value["value"]
            else:
                _collect_yaml_parameters(value, parameters)
        else:
            parameters[key] = value


def _single_component_batch_params(
    *,
    ys1: float,
    ys2: float,
    n: float,
    re: float,
    q: float,
    pa: float,
    flux: float,
    pix_scale: float,
    rmaxf: float,
) -> SersicBatchParams:
    catalog = {
        "reff_arcsec": np.asarray([re], dtype=float),
        "sersic_index": np.asarray([n], dtype=float),
        "axis_ratio": np.asarray([q], dtype=float),
        "position_angle": np.asarray([pa], dtype=float),
        "pos_x_arcsec": np.asarray([ys1], dtype=float),
        "pos_y_arcsec": np.asarray([ys2], dtype=float),
    }
    return _build_batch_params(catalog=catalog, flux=np.asarray([flux], dtype=float), pix_scale=pix_scale, rmaxf=rmaxf)


if njit is not None:

    @njit(cache=False, fastmath=True, nogil=True)
    def _render_sersic_chunk(
        y1: np.ndarray,
        y2: np.ndarray,
        ys1: np.ndarray,
        ys2: np.ndarray,
        amp: np.ndarray,
        bn: np.ndarray,
        inv_re: np.ndarray,
        inv_n: np.ndarray,
        cos_pa: np.ndarray,
        sin_pa: np.ndarray,
        inv_q: np.ndarray,
        rmax: float,
    ) -> np.ndarray:
        out = np.zeros(y1.shape, dtype=np.float64)
        y1_flat = y1.ravel()
        y2_flat = y2.ravel()
        out_flat = out.ravel()
        ncl = ys1.size
        for i in range(y1_flat.size):
            v = 0.0
            y1v = y1_flat[i]
            y2v = y2_flat[i]
            for j in range(ncl):
                dx = y1v - ys1[j]
                dy = y2v - ys2[j]
                x = cos_pa[j] * dx + sin_pa[j] * dy
                yy = -sin_pa[j] * dx + cos_pa[j] * dy
                r = np.sqrt((x * inv_q[j]) * (x * inv_q[j]) + yy * yy)
                if rmax > 0.0 and r >= rmax:
                    continue
                v += amp[j] * np.exp(-bn[j] * ((r * inv_re[j]) ** inv_n[j] - 1.0))
            out_flat[i] = v
        return out



def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_eval(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(elt) for elt in node.elts)
    if isinstance(node, ast.Set):
        return {_safe_eval(elt) for elt in node.elts}
    if isinstance(node, ast.Dict):
        return {_safe_eval(key): _safe_eval(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        if node.id == "np":
            return np
        if node.id == "pi":
            return np.pi
        if node.id == "e":
            return np.e
        raise ValueError(f"Unsupported name: {node.id}")
    if isinstance(node, ast.Attribute):
        value = _safe_eval(node.value)
        if value is np and node.attr in {"pi", "e"}:
            return getattr(np, node.attr)
        raise ValueError(f"Unsupported attribute: {node.attr}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        return +value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise ValueError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")


def _parse_value(value_part: str) -> Any:
    value_part = value_part.strip()
    try:
        return ast.literal_eval(value_part)
    except Exception:
        pass
    try:
        return _safe_eval(ast.parse(value_part, mode="eval").body)
    except Exception:
        return value_part


@lru_cache(maxsize=None)
def load_input_parameters(file_name: str) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {}
    file_path = Path(file_name).expanduser().resolve()
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        with open(file_path, "r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a mapping in YAML input file: {file_path}")
        _collect_yaml_parameters(loaded, parameters)
        for key, value in list(parameters.items()):
            if key.startswith("PATH_TO_") and isinstance(value, str):
                path_value = Path(value).expanduser()
                if not path_value.is_absolute():
                    path_value = (file_path.parent / path_value).resolve()
                parameters[key] = str(path_value)
        return parameters

    with open(file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "#" not in line:
                continue

            value_part, description = line.split("#", 1)
            key = _extract_parameter_key(description)
            if not key:
                continue
            value = _parse_value(value_part)
            if key.startswith("PATH_TO_") and isinstance(value, str):
                path_value = Path(value).expanduser()
                if not path_value.is_absolute():
                    path_value = (file_path.parent / path_value).resolve()
                value = str(path_value)
            parameters[key] = value
    return parameters


def _ensure_parent_dir(file_path: str) -> None:
    Path(file_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _format_value(value: Any) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.8g}"
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _write_table(file_path: str, headers: Sequence[str], columns: Sequence[Sequence[Any]]) -> None:
    _ensure_parent_dir(file_path)
    rows = zip(*columns)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(" ".join(headers) + "\n")
        for row in rows:
            file.write(" ".join(_format_value(value) for value in row) + "\n")


def _write_key_value(file_path: str, key: str, value: Any) -> None:
    _ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"{key}: {_format_value(value)}\n")


def _load_table(file_path: str):
    data = np.genfromtxt(file_path, names=True, dtype=None, encoding="utf-8")
    return np.atleast_1d(data)


def _should_show_plots() -> bool:
    return os.environ.get("CLUMPYLEN_SHOW_PLOTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _is_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prompt_choice(prompt: str, options: Sequence[str], default: Optional[str] = None) -> str:
    print(prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    while True:
        suffix = f" [default: {default}]" if default is not None else ""
        choice = input(f"Enter choice number{suffix}: ").strip()
        if not choice and default is not None:
            return default
        try:
            index = int(choice) - 1
        except ValueError:
            print("Please enter a valid number.")
            continue
        if 0 <= index < len(options):
            return options[index]
        print("Choice out of range.")


def _prompt_yes_no(prompt: str, default: str = "yes") -> str:
    return _prompt_choice(prompt, ["yes", "no"], default=default)


@lru_cache(maxsize=None)
def _load_sed(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, usecols=(0, 1))
    return data[:, 0], data[:, 1]


@lru_cache(maxsize=None)
def _load_ygg_table(ygg_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(ygg_file)
    return data[:, 0], data[:, 1], data


def _nearest_band_flux_per_mass(
    sed_path: str,
    redshift: float,
    band_wavelength: float,
    dl_pc: float,
    mass_ref: float,
) -> float:
    lam0, lum = _load_sed(sed_path)
    lam = lam0 * (1.0 + redshift)
    flux = float(np.interp(band_wavelength, lam, lum)) / (4.0 * np.pi * dl_pc**2)
    return flux / mass_ref


def _nearest_band_mag_ref(
    ygg_file: str,
    age_key: str,
    filter_idx: int,
    telescope_name: str,
) -> Tuple[float, float]:
    data = np.loadtxt(ygg_file)
    ages = data[:, 0]
    masses = data[:, 1]
    mag_col = YGG_MAG_COLUMNS[telescope_name][filter_idx]
    mag_ref = data[:, mag_col]

    age_lookup = {f"{float(age):.0e}": idx for idx, age in enumerate(ages)}
    if age_key not in age_lookup:
        raise ValueError(f"Age {age_key} was not found in {ygg_file}")
    idx = age_lookup[age_key]
    return float(masses[idx]), float(mag_ref[idx])


def _cluster_age_key(age: Any) -> str:
    return f"{float(age):.0e}"


def _project_disk_axis_ratio(q_disk: float, inclination_deg: float) -> float:
    inclination_rad = np.deg2rad(np.clip(float(inclination_deg), 0.0, 90.0))
    projected_q = float(q_disk) * float(np.cos(inclination_rad))
    return float(np.clip(projected_q, 1.0e-2, 1.0))


def _perlin_parameters(cfg: Dict[str, Any]) -> Tuple[float, int, float, float, int]:
    scale = float(cfg.get("perlin_scale", 10.0))
    octaves = max(1, int(cfg.get("perlin_octaves", 3)))
    persistence = float(cfg.get("perlin_persistence", 0.5))
    lacunarity = float(cfg.get("perlin_lacunarity", 2.0))
    seed_value = cfg.get("perlin_seed", 42)
    seed = 42 if seed_value in (None, "") else int(seed_value)
    return scale, octaves, persistence, lacunarity, seed


def _spiral_parameters(cfg: Dict[str, Any], defaults: Tuple[float, float, float]) -> Tuple[float, float, float]:
    spiral_A = float(cfg.get("spiral_A", defaults[0]))
    spiral_alpha = float(cfg.get("spiral_alpha", defaults[1]))
    spiral_Phid = float(cfg.get("spiral_Phid", defaults[2]))
    return spiral_A, spiral_alpha, spiral_Phid


def _apply_spiral(
    image_lens: np.ndarray,
    image_unl: np.ndarray,
    *,
    grid_cache: RenderGridCache,
    gal_x: float,
    gal_y: float,
    n: float,
    re: float,
    pa: float,
    flux: float,
    A: float,
    Na: float,
    Phid: float,
    alpha: float,
    phi: float,
    z0: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    lens_npix = int(image_lens.shape[0])
    unl_npix = int(image_unl.shape[0])
    px = grid_cache.pix_scale
    px_unl = grid_cache.pix_scale_unl

    lens_modifier = _resample_field(
        spiral_modifier_from_grid(
            grid_cache.lensed_y1,
            grid_cache.lensed_y2,
            gal_x,
            gal_y,
            pa,
            phi,
            A,
            Na,
            Phid,
            alpha,
            re,
        ),
        image_lens.shape,
    )
    brightness_convolved = np.asarray(image_lens, dtype=float) * lens_modifier
    lens_sum = float(np.sum(image_lens))
    current_sum = float(np.sum(brightness_convolved))
    if current_sum > 0.0:
        brightness_convolved *= lens_sum / current_sum
    if z0 > 0.0:
        brightness_convolved = gaussian_filter(brightness_convolved, z0 / px)
        current_sum = float(np.sum(brightness_convolved))
        if current_sum > 0.0:
            brightness_convolved *= lens_sum / current_sum

    unl_modifier = _resample_field(
        spiral_modifier_from_grid(
            grid_cache.unlensed_y1,
            grid_cache.unlensed_y2,
            gal_x,
            gal_y,
            pa,
            phi,
            A,
            Na,
            Phid,
            alpha,
            re,
        ),
        image_unl.shape,
    )
    brightness_unl = np.asarray(image_unl, dtype=float) * unl_modifier
    unl_sum = float(np.sum(image_unl))
    current_sum = float(np.sum(brightness_unl))
    if current_sum > 0.0:
        brightness_unl *= unl_sum / current_sum
    if z0 > 0.0:
        brightness_unl = gaussian_filter(brightness_unl, z0 / px_unl)
        current_sum = float(np.sum(brightness_unl))
        if current_sum > 0.0:
            brightness_unl *= unl_sum / current_sum
    return brightness_convolved, brightness_unl


def _apply_perlin(
    image_lens: np.ndarray,
    image_unl: np.ndarray,
    *,
    scale: float = 10.0,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    lens_noise = generate_perlin_noise(
        image_lens.shape[1],
        image_lens.shape[0],
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        seed=seed,
    )
    unl_noise = generate_perlin_noise(
        image_unl.shape[1],
        image_unl.shape[0],
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        seed=seed,
    )

    lens = image_lens * lens_noise
    unl = image_unl * unl_noise

    lens_sum = float(np.sum(lens))
    unl_sum = float(np.sum(unl))
    if lens_sum > 0:
        lens = lens / lens_sum * float(np.sum(image_lens))
    if unl_sum > 0:
        unl = unl / unl_sum * float(np.sum(image_unl))
    return lens, unl


def _build_component(
    *,
    size: float,
    npix: int,
    df,
    sizex: Sequence[float],
    sizey: Sequence[float],
    size_unl: float,
    npix_unl: int,
    gal_x: float,
    gal_y: float,
    flux: float,
    z_s: float,
    n: float,
    re: float,
    q: float,
    pa: float,
    rmaxf: float,
) -> SersicModel:
    kwargs = {
        "n": n,
        "re": re,
        "q": q,
        "pa": pa,
        "ys1": gal_x,
        "ys2": gal_y,
        "flux": flux,
        "zs": z_s,
    }
    return SersicModel(
        size=size,
        Npix=npix,
        gl=df,
        sizex=sizex,
        sizey=sizey,
        save_unlensed=True,
        size_unlensed=size_unl,
        y1_unlensed=gal_x,
        y2_unlensed=gal_y,
        npix_unlensed=npix_unl,
        save_unlensed_recenter=True,
        rmaxf=rmaxf,
        **kwargs,
    )


def _render_clumps(
    *,
    clump_fluxes: np.ndarray,
    clump_params: Dict[str, np.ndarray],
    size: float,
    npix: int,
    df,
    sizex: Sequence[float],
    sizey: Sequence[float],
    size_unl: float,
    npix_unl: int,
    gal_x: float,
    gal_y: float,
    z_s: float,
    rmaxf: float,
    compute_point_sources: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    se_total = np.zeros((npix, npix), dtype=float)
    se_unl_total = np.zeros((npix_unl, npix_unl), dtype=float)
    xi_parts: List[np.ndarray] = []
    yi_parts: List[np.ndarray] = []
    mui_parts: List[np.ndarray] = []
    mui_tg_parts: List[np.ndarray] = []

    for i in tqdm(range(int(clump_params["N_cl"])), desc="Rendering clumps"):
        component = _build_component(
            size=size,
            npix=npix,
            df=df,
            sizex=sizex,
            sizey=sizey,
            size_unl=size_unl,
            npix_unl=npix_unl,
            gal_x=gal_x,
            gal_y=gal_y,
            flux=float(clump_fluxes[i]),
            z_s=z_s,
            n=float(clump_params["sersic_index"][i]),
            re=float(clump_params["reff_arcsec"][i]),
            q=float(clump_params["axis_ratio"][i]),
            pa=float(clump_params["position_angle"][i]),
            rmaxf=rmaxf,
        )
        se_total += component.image
        se_unl_total += component.image_unlensed

        if compute_point_sources:
            kwargs = {
                "ys1": float(clump_params["pos_x_arcsec"][i]),
                "ys2": float(clump_params["pos_y_arcsec"][i]),
                "flux": float(clump_fluxes[i]),
                "zs": z_s,
            }
            ps = pointsrc(size=size, sizex=sizex, sizey=sizey, Npix=npix, gl=df, **kwargs)
            xi, yi, mui, mui_tg = ps.find_images()
            xi_parts.append(np.asarray(xi))
            yi_parts.append(np.asarray(yi))
            mui_parts.append(np.asarray(mui))
            mui_tg_parts.append(np.asarray(mui_tg))

    return se_total, se_unl_total, xi_parts, yi_parts, mui_parts, mui_tg_parts


def _deposit_bilinear(
    canvas: np.ndarray,
    x: float,
    y: float,
    flux: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(flux) or flux == 0.0:
        return
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return

    nx = canvas.shape[1]
    ny = canvas.shape[0]
    px = (x - x_min) / (x_max - x_min) * (nx - 1)
    py = (y - y_min) / (y_max - y_min) * (ny - 1)
    ix = int(np.floor(px))
    iy = int(np.floor(py))
    wx = px - ix
    wy = py - iy

    for dx, wxw in ((0, 1.0 - wx), (1, wx)):
        xx = ix + dx
        if xx < 0 or xx >= nx:
            continue
        for dy, wyw in ((0, 1.0 - wy), (1, wy)):
            yy = iy + dy
            if yy < 0 or yy >= ny:
                continue
            canvas[yy, xx] += flux * wxw * wyw


def _soften_sparse_canvas(canvas: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    total = float(np.sum(canvas))
    if total <= 0.0:
        return canvas
    blurred = gaussian_filter(canvas, sigma=sigma, mode="constant")
    current = float(np.sum(blurred))
    if current > 0.0:
        blurred *= total / current
    return blurred


def _render_clump_point_sources(
    *,
    catalog: Dict[str, Any],
    fluxes: np.ndarray,
    df,
    size: float,
    npix: int,
    sizex: Sequence[float],
    sizey: Sequence[float],
    size_unl: float,
    npix_unl: int,
    gal_x: float,
    gal_y: float,
    z_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render unresolved clumps as sparse point-source stamps.

    The clumps in the demo are much smaller than one image pixel, so sampling
    them as extended Sersic profiles on a coarse grid can erase them entirely.
    This path keeps them visible without forcing the full galaxy render to run
    at the same resolution.
    """

    se_total = np.zeros((npix, npix), dtype=float)
    se_unl_total = np.zeros((npix_unl, npix_unl), dtype=float)
    if len(fluxes) == 0:
        return se_total, se_unl_total

    point_x, point_y, point_mu, _ = _collect_point_source_catalog(
        cluster="",
        path_clumps="",
        catalog=catalog,
        df=df,
        size=size,
        sizex=sizex,
        sizey=sizey,
        npix=npix,
        z_s=z_s,
    )

    x_min, x_max = float(sizex[0]), float(sizex[1])
    y_min, y_max = float(sizey[0]), float(sizey[1])
    ux_min = float(gal_x - size_unl / 2.0)
    ux_max = float(gal_x + size_unl / 2.0)
    uy_min = float(gal_y - size_unl / 2.0)
    uy_max = float(gal_y + size_unl / 2.0)

    source_x = np.asarray(catalog["pos_x_arcsec"], dtype=float)
    source_y = np.asarray(catalog["pos_y_arcsec"], dtype=float)

    for i, flux in enumerate(np.asarray(fluxes, dtype=float)):
        if not np.isfinite(flux) or flux == 0.0:
            continue

        # Lensed point-source images.
        if i < len(point_x):
            xs = np.asarray(point_x[i], dtype=float)
            ys = np.asarray(point_y[i], dtype=float)
            mus = np.asarray(point_mu[i], dtype=float)
            for x, y, mu in zip(xs, ys, mus):
                _deposit_bilinear(
                    se_total,
                    float(x),
                    float(y),
                    flux * abs(float(mu)),
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                )

        # Unlensed source-plane point.
        if i < source_x.size:
            _deposit_bilinear(
                se_unl_total,
                float(source_x[i]),
                float(source_y[i]),
                flux,
                ux_min,
                ux_max,
                uy_min,
                uy_max,
            )

    se_total = _soften_sparse_canvas(se_total, sigma=0.6)
    se_unl_total = _soften_sparse_canvas(se_unl_total, sigma=0.6)
    return se_total, se_unl_total


def _format_header(hdu: fits.PrimaryHDU, ra_c: float, dec_c: float, pix: float, npix: int) -> None:
    header = hdu.header
    header["RADESYS"] = "ICRS"
    header["CTYPE1"] = "RA---TAN"
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT2"] = "deg"
    header["CD1_1"] = -pix
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = pix
    header["NAXIS"] = 2
    header["CRPIX1"] = (npix + 1) / 2.0
    header["CRPIX2"] = (npix + 1) / 2.0
    header["CRVAL1"] = ra_c
    header["CRVAL2"] = dec_c
    header["NAXIS1"] = npix
    header["NAXIS2"] = npix


def _source_rescaling_factor(df, z_s: float) -> float:
    if df is None:
        return 1.0
    lens_z = float(getattr(df, "zl", z_s))
    src_z = float(getattr(df, "zs", z_s))
    if np.isclose(z_s, src_z):
        return 1.0
    if z_s > lens_z:
        ds = float(df.co.angular_diameter_distance(z_s).value)
        dls = float(df.co.angular_diameter_distance_z1z2(lens_z, z_s).value)
        if ds > 0.0 and float(getattr(df, "dls", 0.0)) > 0.0:
            return dls / ds * float(df.ds) / float(df.dls)
    return 0.0


def _ray_trace_grid(df, x1: np.ndarray, x2: np.ndarray, z_s: float) -> Tuple[np.ndarray, np.ndarray]:
    px = df.pixel_scale
    x1pix = (x1 - df.thetax[0]) / px
    x2pix = (x2 - df.thetay[0]) / px

    if len(x1pix.shape) > 1:
        x1pix[:, -1] = np.round(x1pix[:, -1], 0)
        x2pix[-1, :] = np.round(x2pix[-1, :], 0)
        x1pix[:, 0] = np.round(x1pix[:, 0], 0)
        x2pix[0, :] = np.round(x2pix[0, :], 0)
    else:
        x1pix[-1] = np.round(x1pix[-1], 0)
        x2pix[-1] = np.round(x2pix[-1], 0)
        x1pix[0] = np.round(x1pix[0], 0)
        x2pix[0] = np.round(x2pix[0], 0)

    a1 = map_coordinates(df.a1, [x2pix, x1pix], order=1, prefilter=True)
    a2 = map_coordinates(df.a2, [x2pix, x1pix], order=1, prefilter=True)
    rescf = _source_rescaling_factor(df, z_s)

    if getattr(df, "perturbed", False):
        a11 = 1.0 - df.pb.pkappa - df.pb.pgamma1
        a22 = 1.0 - df.pb.pkappa + df.pb.pgamma1
        a12 = -df.pb.pgamma2
        a111 = -0.5 * (df.pb.pg1 + 3.0 * df.pb.pf1)
        a222 = -0.5 * (3.0 * df.pb.pf2 - df.pb.pg2)
        a112 = -0.5 * (df.pb.pf2 + df.pb.pg2)
        a221 = -0.5 * (df.pb.pf1 - df.pb.pg1)

        z1 = x1 - a1 * rescf - df.pb.pa1 * rescf
        z2 = x2 - a2 * rescf - df.pb.pa2 * rescf

        k1 = a11 * z1 + a12 * z2 + 0.5 * a111 * z1**2 + a112 * z1 * z2 + 0.5 * a221 * z2**2
        k2 = a22 * z2 + a12 * z1 + 0.5 * a222 * z2**2 + a221 * z1 * z2 + 0.5 * a112 * z1**2
        y1 = k1
        y2 = k2
    else:
        y1 = x1 - a1 * rescf
        y2 = x2 - a2 * rescf

    return y1, y2


def _build_render_grid_cache(
    *,
    df,
    z_s: float,
    size: float,
    npix: int,
    size_unl: float,
    npix_unl: int,
    sizex: Optional[Sequence[float]] = None,
    sizey: Optional[Sequence[float]] = None,
    gal_x: float,
    gal_y: float,
    render_scale: float = 1.0,
) -> RenderGridCache:
    render_scale = float(render_scale)
    if render_scale <= 0.0:
        raise ValueError("render_scale must be positive")

    npix_lensed = max(8, int(round(npix * render_scale)))
    npix_unlensed = max(8, int(round(npix_unl * render_scale)))

    pix_scale = size / max(1, npix_lensed - 1)
    pix_scale_unl = size_unl / max(1, npix_unlensed - 1)

    # The lensed canvas must stay in the observation frame defined by the
    # lens-plane bounds. The source offset belongs only in the brightness
    # evaluation, not in the image-plane grid.
    if sizex is None:
        x = np.linspace(-size / 2.0, size / 2.0, npix_lensed)
    else:
        x = np.linspace(float(sizex[0]), float(sizex[1]), npix_lensed)
    if sizey is None:
        y = np.linspace(-size / 2.0, size / 2.0, npix_lensed)
    else:
        y = np.linspace(float(sizey[0]), float(sizey[1]), npix_lensed)
    x1, x2 = np.meshgrid(x, y)
    y1, y2 = _ray_trace_grid(df, x1, x2, z_s)

    xu = np.linspace(-size_unl / 2.0, size_unl / 2.0, npix_unlensed) + gal_x
    yu = np.linspace(-size_unl / 2.0, size_unl / 2.0, npix_unlensed) + gal_y
    x1u, x2u = np.meshgrid(xu, yu)
    y1u, y2u = x1u, x2u

    return RenderGridCache(
        lensed_y1=np.asarray(y1, dtype=float),
        lensed_y2=np.asarray(y2, dtype=float),
        unlensed_y1=np.asarray(y1u, dtype=float),
        unlensed_y2=np.asarray(y2u, dtype=float),
        lensed_shape=y1.shape,
        unlensed_shape=y1u.shape,
        pix_scale=pix_scale,
        pix_scale_unl=pix_scale_unl,
    )


def _slice_batch_params(params: SersicBatchParams, sl: slice) -> SersicBatchParams:
    return SersicBatchParams(
        ys1=params.ys1[sl],
        ys2=params.ys2[sl],
        re=params.re[sl],
        n=params.n[sl],
        q=params.q[sl],
        pa=params.pa[sl],
        flux=params.flux[sl],
        amp=params.amp[sl],
        bn=params.bn[sl],
        inv_re=params.inv_re[sl],
        inv_n=params.inv_n[sl],
        cos_pa=params.cos_pa[sl],
        sin_pa=params.sin_pa[sl],
        inv_q=params.inv_q[sl],
        rmax=params.rmax,
    )


def _estimate_render_workers(n_cl: int, grid_cache: RenderGridCache, memory_budget_mb: float = 512.0) -> Tuple[int, int]:
    if n_cl <= 1:
        return 1, 1

    canvas_bytes = grid_cache.lensed_y1.nbytes + grid_cache.unlensed_y1.nbytes
    per_worker_mb = max(1.0, (2.0 * canvas_bytes) / (1024.0**2))
    worker_cap = os.cpu_count() or 1
    workers = max(1, min(worker_cap, n_cl, int(memory_budget_mb // per_worker_mb) or 1))
    chunk_size = max(1, int(np.ceil(n_cl / workers)))
    return workers, chunk_size


def _render_batch_worker(
    *,
    grid_cache: RenderGridCache,
    params: SersicBatchParams,
    sl: slice,
    df=None,
    size: float = 0.0,
    npix: int = 0,
    sizex: Optional[Sequence[float]] = None,
    sizey: Optional[Sequence[float]] = None,
    size_unl: float = 0.0,
    npix_unl: int = 0,
    gal_x: float = 0.0,
    gal_y: float = 0.0,
    z_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    sub = _slice_batch_params(params, sl)
    if njit is not None:
        se = _render_sersic_chunk(
            grid_cache.lensed_y1,
            grid_cache.lensed_y2,
            sub.ys1,
            sub.ys2,
            sub.amp,
            sub.bn,
            sub.inv_re,
            sub.inv_n,
            sub.cos_pa,
            sub.sin_pa,
            sub.inv_q,
            sub.rmax,
        )
        se_unl = _render_sersic_chunk(
            grid_cache.unlensed_y1,
            grid_cache.unlensed_y2,
            sub.ys1,
            sub.ys2,
            sub.amp,
            sub.bn,
            sub.inv_re,
            sub.inv_n,
            sub.cos_pa,
            sub.sin_pa,
            sub.inv_q,
            sub.rmax,
        )
        return se, se_unl

    # Numba fallback: keep the same geometry cache, but use the slower
    # Python-side Sersic model only if compiled kernels are unavailable.
    se = np.zeros(grid_cache.lensed_shape, dtype=float)
    se_unl = np.zeros(grid_cache.unlensed_shape, dtype=float)
    for i in range(sub.ys1.size):
        component = _build_component(
            size=size,
            npix=grid_cache.lensed_shape[0],
            df=df,
            sizex=sizex or (-size / 2.0, size / 2.0),
            sizey=sizey or (-size / 2.0, size / 2.0),
            size_unl=size_unl,
            npix_unl=grid_cache.unlensed_shape[0],
            gal_x=gal_x,
            gal_y=gal_y,
            flux=float(sub.flux[i]),
            z_s=z_s,
            n=float(sub.n[i]),
            re=float(sub.re[i]),
            q=float(sub.q[i]),
            pa=float(sub.pa[i]),
            rmaxf=sub.rmax,
        )
        se += component.image
        se_unl += component.image_unlensed
    return se, se_unl


def _render_batch_parallel(
    *,
    grid_cache: RenderGridCache,
    params: SersicBatchParams,
    df=None,
    size: float = 0.0,
    npix: int = 0,
    sizex: Optional[Sequence[float]] = None,
    sizey: Optional[Sequence[float]] = None,
    size_unl: float = 0.0,
    npix_unl: int = 0,
    gal_x: float = 0.0,
    gal_y: float = 0.0,
    z_s: float = 1.0,
    memory_budget_mb: float = 512.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cl = int(params.ys1.size)
    if n_cl == 0:
        return np.zeros(grid_cache.lensed_shape, dtype=float), np.zeros(grid_cache.unlensed_shape, dtype=float)

    workers, chunk_size = _estimate_render_workers(n_cl, grid_cache, memory_budget_mb=memory_budget_mb)
    slices = [slice(start, min(start + chunk_size, n_cl)) for start in range(0, n_cl, chunk_size)]

    if workers == 1 or len(slices) == 1:
        return _render_batch_worker(
            grid_cache=grid_cache,
            params=params,
            sl=slices[0],
            df=df,
            size=size,
            npix=npix,
            sizex=sizex,
            sizey=sizey,
            size_unl=size_unl,
            npix_unl=npix_unl,
            gal_x=gal_x,
            gal_y=gal_y,
            z_s=z_s,
        )

    se_total = np.zeros(grid_cache.lensed_shape, dtype=float)
    se_unl_total = np.zeros(grid_cache.unlensed_shape, dtype=float)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _render_batch_worker,
                grid_cache=grid_cache,
                params=params,
                sl=sl,
                df=df,
                size=size,
                npix=npix,
                sizex=sizex,
                sizey=sizey,
                size_unl=size_unl,
                npix_unl=npix_unl,
                gal_x=gal_x,
                gal_y=gal_y,
                z_s=z_s,
            )
            for sl in slices
        ]
        for fut in futures:
            se, se_unl = fut.result()
            se_total += se
            se_unl_total += se_unl
    return se_total, se_unl_total


def _resample_flux_preserving(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if image.shape == target_shape:
        return image
    zoom_y = target_shape[0] / image.shape[0]
    zoom_x = target_shape[1] / image.shape[1]
    resized = zoom(image, (zoom_y, zoom_x), order=1)
    resized = resized[: target_shape[0], : target_shape[1]]
    if resized.shape != target_shape:
        out = np.zeros(target_shape, dtype=resized.dtype)
        ny = min(target_shape[0], resized.shape[0])
        nx = min(target_shape[1], resized.shape[1])
        out[:ny, :nx] = resized[:ny, :nx]
        resized = out
    original = float(np.sum(image))
    current = float(np.sum(resized))
    if current > 0.0:
        resized = resized * (original / current)
    return resized


def _resample_field(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if image.shape == target_shape:
        return image
    zoom_y = target_shape[0] / image.shape[0]
    zoom_x = target_shape[1] / image.shape[1]
    resized = zoom(image, (zoom_y, zoom_x), order=1)
    resized = resized[: target_shape[0], : target_shape[1]]
    if resized.shape != target_shape:
        out = np.zeros(target_shape, dtype=resized.dtype)
        ny = min(target_shape[0], resized.shape[0])
        nx = min(target_shape[1], resized.shape[1])
        out[:ny, :nx] = resized[:ny, :nx]
        resized = out
    return resized


def _render_single_component_from_cache(
    *,
    grid_cache: RenderGridCache,
    size: float,
    npix: int,
    size_unl: float,
    npix_unl: int,
    sizex: Optional[Sequence[float]],
    sizey: Optional[Sequence[float]],
    gal_x: float,
    gal_y: float,
    ys1: float,
    ys2: float,
    flux: float,
    z_s: float,
    n: float,
    re: float,
    q: float,
    pa: float,
    rmaxf: float,
    render_memory_mb: float,
) -> Tuple[np.ndarray, np.ndarray]:
    catalog = {
        "reff_arcsec": np.asarray([re], dtype=float),
        "sersic_index": np.asarray([n], dtype=float),
        "axis_ratio": np.asarray([q], dtype=float),
        "position_angle": np.asarray([pa], dtype=float),
        "pos_x_arcsec": np.asarray([ys1], dtype=float),
        "pos_y_arcsec": np.asarray([ys2], dtype=float),
    }
    params = _build_batch_params(catalog=catalog, flux=np.asarray([flux], dtype=float), pix_scale=grid_cache.pix_scale, rmaxf=rmaxf)
    se, se_unl = _render_batch_parallel(
        grid_cache=grid_cache,
        params=params,
        size=size,
        npix=grid_cache.lensed_shape[0],
        sizex=sizex,
        sizey=sizey,
        size_unl=size_unl,
        npix_unl=grid_cache.unlensed_shape[0],
        gal_x=gal_x,
        gal_y=gal_y,
        z_s=z_s,
        memory_budget_mb=render_memory_mb,
    )
    if se.shape != (npix, npix):
        se = _resample_flux_preserving(se, (npix, npix))
    if se_unl.shape != (npix_unl, npix_unl):
        se_unl = _resample_flux_preserving(se_unl, (npix_unl, npix_unl))
    return se, se_unl


def _sample_mass_dist_sch(
    *,
    alpha: float = 2.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    xstar: float = 8.5e12,
    xmin: float = 1e4,
    xmax: float = 1e13,
    masstotlim: float = 1e8,
    nrx: int = 999,
) -> np.ndarray:
    if masstotlim <= 0:
        return np.asarray([], dtype=float)

    lxmin = np.log10(xmin)
    lxmax = np.log10(xmax)
    lx = np.linspace(lxmin, lxmax, nrx)
    xtmp = 10.0**lx

    pdf = xtmp ** (-alpha) * np.exp(-delta * (xtmp / xstar) ** gamma)
    shell_width = np.gradient(xtmp)
    weights = pdf * shell_width
    cdf = np.cumsum(weights)
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError("Invalid mass distribution normalization.")
    cdf /= cdf[-1]

    mean_mass = float(np.sum(xtmp * weights) / np.sum(weights))
    masses: List[np.ndarray] = []
    total_mass = 0.0
    while total_mass < masstotlim:
        remaining = masstotlim - total_mass
        batch = max(512, int(np.ceil(remaining / max(mean_mass, 1.0) * 1.5)))
        cran = np.random.random(batch)
        sampled = np.interp(cran, cdf, xtmp)
        masses.append(sampled)
        total_mass += float(sampled.sum())

    return np.concatenate(masses).astype(float, copy=False)


def _sample_ages(alpha_t: float, t_min: float, t_max: float, n_cl: int) -> Tuple[np.ndarray, np.ndarray]:
    nrx = 999
    ltmin = np.log10(t_min)
    ltmax = np.log10(t_max)
    lx = np.linspace(ltmin, ltmax, nrx)
    xtmp = 10.0**lx
    weights = xtmp ** (-alpha_t)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    ages = xtmp[np.searchsorted(cdf, np.random.random(n_cl), side="left")]
    ages_cl = np.char.mod("%.0e", ages)
    return ages, ages_cl


def _make_clump_catalog(
    *,
    npix_unl: int,
    size_unl: float,
    seGAL_unl: np.ndarray,
    dl_pc: Distance,
    zs: float,
    gal_x: float,
    gal_y: float,
    fsub: float,
    mass_bulge: float,
    mass_disk: float,
    mass_total: float,
    alpha: float,
    gamma: float,
    delta: float,
    mass_min: float,
    mass_max: float,
    trunc_mass: float,
    alpha_t: float,
    t_min: float,
    t_max: float,
    co,
) -> Dict[str, Any]:
    mass_clumps = mass_disk * fsub
    print(
        "Mass chosen: total:",
        f"{mass_total:e}",
        "; bulge:",
        f"{mass_bulge:e}",
        "; disc:",
        f"{mass_disk:e}",
        "; clumps:",
        f"{mass_clumps:e}",
        "M_sun",
    )

    mass_cl = _sample_mass_dist_sch(
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        xstar=trunc_mass,
        xmin=mass_min,
        xmax=mass_max,
        masstotlim=mass_clumps,
    )
    n_cl = int(len(mass_cl))
    print(f"Simulating {n_cl} clumps.")

    r_eff_mean = 2.55 * (mass_cl / 1.0e4) ** 0.24
    sigma = 0.3
    r_eff = r_eff_mean * np.random.lognormal(0.0, sigma, n_cl) * u.pc
    d_a = co.angular_diameter_distance(zs)
    reff_arcsec = (r_eff / d_a).to(u.arcsec, u.dimensionless_angles()).value

    pos_x, pos_y = sample2Dimage(seGAL_unl.T, n=n_cl)
    pos_x_arcsec = (pos_x - npix_unl / 2.0) * (size_unl / npix_unl) + gal_x
    pos_y_arcsec = (pos_y - npix_unl / 2.0) * (size_unl / npix_unl) + gal_y

    se_ind = np.full(n_cl, 0.5, dtype=float)
    pos_ang = np.random.rand(n_cl) * np.pi
    ax_rat = np.random.rand(n_cl) * 0.7 + 0.3
    ages, ages_cl = _sample_ages(alpha_t=alpha_t, t_min=t_min, t_max=t_max, n_cl=n_cl)

    return {
        "N_cl": n_cl,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "pos_x_arcsec": pos_x_arcsec,
        "pos_y_arcsec": pos_y_arcsec,
        "reff_arcsec": reff_arcsec,
        "reff_pc": np.asarray(r_eff.value, dtype=float),
        "sersic_index": se_ind,
        "position_angle": pos_ang,
        "axis_ratio": ax_rat,
        "mass_cl": mass_cl,
        "ages_cl": ages_cl,
        "ages": ages,
    }


def _save_clump_catalog(cluster: str, path_clumps: str, catalog: Dict[str, Any]) -> None:
    cluster_dir = Path(path_clumps) / cluster
    cluster_dir.mkdir(parents=True, exist_ok=True)
    param_file = cluster_dir / "clumps_parameters_column.txt"
    _write_table(
        str(param_file),
        [
            "ID",
            "Sersic_Index",
            "Effective_Radius_arcsec",
            "Effective_Radius_pc",
            "Axis_Ratio",
            "Position_Angle",
            "Pos_x_arcsec",
            "Pos_y_arcsec",
            "Stellar_Mass",
            "Age_yr",
        ],
        [
            np.arange(catalog["N_cl"], dtype=int),
            catalog["sersic_index"],
            np.asarray(catalog["reff_arcsec"], dtype=float),
            np.asarray(catalog["reff_pc"], dtype=float),
            catalog["axis_ratio"],
            catalog["position_angle"],
            catalog["pos_x_arcsec"],
            catalog["pos_y_arcsec"],
            catalog["mass_cl"],
            catalog["ages"],
        ],
    )


def _load_clump_catalog(cluster: str, path_clumps: str) -> Dict[str, Any]:
    file_path = Path(path_clumps) / cluster / "clumps_parameters_column.txt"
    data = _load_table(str(file_path))
    catalog = {
        "N_cl": len(data),
        "sersic_index": np.asarray(data["Sersic_Index"], dtype=float),
        "reff_arcsec": np.asarray(data["Effective_Radius_arcsec"], dtype=float),
        "reff_pc": np.asarray(data["Effective_Radius_pc"], dtype=float),
        "axis_ratio": np.asarray(data["Axis_Ratio"], dtype=float),
        "position_angle": np.asarray(data["Position_Angle"], dtype=float),
        "pos_x_arcsec": np.asarray(data["Pos_x_arcsec"], dtype=float),
        "pos_y_arcsec": np.asarray(data["Pos_y_arcsec"], dtype=float),
        "mass_cl": np.asarray(data["Stellar_Mass"], dtype=float),
        "ages": np.asarray(data["Age_yr"], dtype=float),
    }
    catalog["ages_cl"] = np.array([f"{age:.0e}" for age in catalog["ages"]])
    return catalog


def _plot_clumps(cluster: str, path_images: str, mass_cl: np.ndarray, reff_pc: np.ndarray, ages: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    cluster_dir = Path(path_images) / cluster
    cluster_dir.mkdir(parents=True, exist_ok=True)

    mass_brown = np.arange(1e2, 1e6, 1e2)
    r_eff_brown = 2.55 * (mass_brown / 1e4) ** 0.24

    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax2.plot(mass_cl, reff_pc, "D", color="magenta")
    ax2.plot(mass_brown, r_eff_brown, "-", color="b")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(7e3, 1e6)
    ax2.set_ylim(5e-1, 5e1)
    ax2.set_ylabel(r"$\rm R_{eff}~[pc]$", fontsize=14)
    ax2.set_xlabel(r"$\rm M_\star~[M_\odot]$", fontsize=14)

    ax1.hist(mass_cl, bins=10 ** np.arange(np.log10(mass_cl.min()), np.log10(mass_cl.max()) + 0.25, 0.25), color="Magenta")
    ax1.set_ylabel(r"$\rm N_{clumps}$", fontsize=17)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(7e3, 1e6)
    ax1.set_ylim(1, 3e2)
    ax1.set_xticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(cluster_dir / "Reff_Mass_with_Hist.pdf", bbox_inches="tight", dpi=300)
    if _should_show_plots():
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax2.plot(ages, mass_cl, "o", color="magenta")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(7e5, 7e8)
    ax2.set_ylabel(r"$\rm M_\star~[M_\odot]$", fontsize=14)
    ax2.set_xlabel(r"$\rm Ages~[yr]$", fontsize=14)

    ax1.hist(ages, bins=10 ** np.arange(np.log10(ages.min()), np.log10(ages.max()) + 0.25, 0.25), color="Magenta")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\rm N_{clumps}$", fontsize=17)
    ax1.set_xlim(7e5, 7e8)
    ax1.set_ylim(1, 2e2)
    ax1.set_xticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(cluster_dir / "Ages_Mass_with_Hist.pdf", bbox_inches="tight", dpi=300)
    if _should_show_plots():
        plt.show()
    plt.close(fig)


def generate_perlin_noise(width, height, scale=10, octaves=3, persistence=0.5, lacunarity=2.0, seed=None):
    if scale <= 0:
        raise ValueError("scale must be positive")

    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _lerp(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        return a + w * (b - a)

    rng = np.random.default_rng(0 if seed is None else int(seed))
    yy, xx = np.meshgrid(np.arange(height, dtype=float), np.arange(width, dtype=float), indexing="ij")
    noise_map = np.zeros((height, width), dtype=float)
    amplitude = 1.0
    total_amplitude = 0.0

    for octave in range(max(1, int(octaves))):
        cell = max(float(scale) / (lacunarity**octave), 1.0)
        gx = int(np.ceil(width / cell)) + 2
        gy = int(np.ceil(height / cell)) + 2
        gradients = rng.normal(size=(gy, gx, 2))
        norms = np.linalg.norm(gradients, axis=-1, keepdims=True)
        gradients = np.divide(gradients, np.where(norms == 0.0, 1.0, norms))

        x = xx / cell
        y = yy / cell
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        xf = x - x0
        yf = y - y0
        x1 = x0 + 1
        y1 = y0 + 1

        g00 = gradients[y0 % gy, x0 % gx]
        g10 = gradients[y0 % gy, x1 % gx]
        g01 = gradients[y1 % gy, x0 % gx]
        g11 = gradients[y1 % gy, x1 % gx]

        d00 = g00[..., 0] * xf + g00[..., 1] * yf
        d10 = g10[..., 0] * (xf - 1.0) + g10[..., 1] * yf
        d01 = g01[..., 0] * xf + g01[..., 1] * (yf - 1.0)
        d11 = g11[..., 0] * (xf - 1.0) + g11[..., 1] * (yf - 1.0)

        u = _fade(xf)
        v = _fade(yf)
        noise_map += amplitude * _lerp(_lerp(d00, d10, u), _lerp(d01, d11, u), v)
        total_amplitude += amplitude
        amplitude *= persistence

    min_val = float(np.min(noise_map))
    max_val = float(np.max(noise_map))
    if np.isclose(min_val, max_val):
        return np.zeros_like(noise_map)
    return (noise_map - min_val) / (max_val - min_val)


def _load_band_flux_scalars(
    *,
    cfg: Dict[str, Any],
    z_s: float,
    dl_pc: float,
    band_wavelength: float,
    band_label: str,
    filter_idx: int,
    telescope_name: str,
    mass_disk: float,
    mass_disk_ref: float,
    mass_bulge: float,
    mass_bulge_ref: float,
) -> Tuple[float, float]:
    path_sed = cfg["PATH_TO_sed"]
    disk_path = str(Path(path_sed) / "sed_arms.sed")
    bulge_path = str(Path(path_sed) / "sed_bulge.sed")
    disk_per_mass = _nearest_band_flux_per_mass(disk_path, z_s, band_wavelength, dl_pc, mass_disk_ref)
    bulge_per_mass = _nearest_band_flux_per_mass(bulge_path, z_s, band_wavelength, dl_pc, mass_bulge_ref)
    return disk_per_mass * mass_disk, bulge_per_mass * mass_bulge


def _reference_band_flux_scalars(
    *,
    cfg: Dict[str, Any],
    z_s: float,
    dl_pc: float,
    band_wavelength: float,
    mass_disk: float,
    mass_disk_ref: float,
    mass_bulge: float,
    mass_bulge_ref: float,
) -> Tuple[float, float]:
    return _load_band_flux_scalars(
        cfg=cfg,
        z_s=z_s,
        dl_pc=dl_pc,
        band_wavelength=band_wavelength,
        band_label="ref",
        filter_idx=0,
        telescope_name="HST_WFC3",
        mass_disk=mass_disk,
        mass_disk_ref=mass_disk_ref,
        mass_bulge=mass_bulge,
        mass_bulge_ref=mass_bulge_ref,
    )


def _instrument_mag_scalars(
    *,
    cfg: Dict[str, Any],
    cluster: str,
    telescope_name: str,
    filter_idx: int,
    ages_cl: np.ndarray,
    mass_cl: np.ndarray,
) -> np.ndarray:
    needed_path = cfg.get("PATH_TO_needed")
    if not needed_path:
        raise ValueError("Instrument mode requires PATH_TO_needed in the input file.")
    ygg_file = str(Path(needed_path) / f"Ygg_{cluster}_{telescope_name}.txt")
    ages, masses, data = _load_ygg_table(ygg_file)
    mag_col = YGG_MAG_COLUMNS[telescope_name][filter_idx]
    mag_ref = data[:, mag_col]
    age_lookup = {f"{float(age):.0e}": idx for idx, age in enumerate(ages)}

    try:
        unique_ages, inverse = np.unique(ages_cl, return_inverse=True)
        unique_indices = np.asarray([age_lookup[age] for age in unique_ages], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Age {exc.args[0]} was not found in {ygg_file}") from exc

    mass_ref_arr = np.asarray(masses[unique_indices], dtype=float)[inverse]
    mag_ref_arr = np.asarray(mag_ref[unique_indices], dtype=float)[inverse]
    mag_cl = mag_ref_arr - 2.5 * np.log10(mass_cl / mass_ref_arr)
    return mag_cl


def _render_multiple_images_catalog(
    *,
    cluster: str,
    path_clumps: str,
    sizex: Sequence[float],
    sizey: Sequence[float],
    ages_cl: np.ndarray,
    reff_pc: np.ndarray,
    mag_cl: np.ndarray,
    point_x: List[np.ndarray],
    point_y: List[np.ndarray],
    point_mu: List[np.ndarray],
    point_mu_tg: List[np.ndarray],
) -> None:
    cluster_dir = Path(path_clumps) / cluster
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if not point_x:
        np.savetxt(cluster_dir / "multiple_images_number.txt", np.asarray([0], dtype=int), fmt="%d", delimiter=",")
        return

    counts = np.asarray([len(xi) for xi in point_x], dtype=int)
    x_tot = np.concatenate(point_x)
    y_tot = np.concatenate(point_y)
    mui_tot = np.concatenate(point_mu)
    mui_tg_tot = np.concatenate(point_mu_tg)

    ages_float = np.asarray([float(age) for age in ages_cl], dtype=float)
    size_imm_mul = np.repeat(np.asarray(reff_pc, dtype=float), counts)
    ages_imm_mul = np.repeat(ages_float, counts)
    mag_imm_mul = np.repeat(np.asarray(mag_cl, dtype=float), counts)

    x_expanded = np.asarray(x_tot, dtype=float)
    y_expanded = np.asarray(y_tot, dtype=float)
    mui_expanded = np.asarray(mui_tot, dtype=float)
    mui_tg_expanded = np.asarray(mui_tg_tot, dtype=float)

    mask = (x_expanded >= sizex[0]) & (x_expanded <= sizex[1]) & (y_expanded >= sizey[0]) & (y_expanded <= sizey[1])
    full_array = np.stack(
        [
            x_expanded[mask],
            y_expanded[mask],
            ages_imm_mul[mask],
            size_imm_mul[mask],
            mui_expanded[mask],
            mui_tg_expanded[mask],
            mag_imm_mul[mask],
        ],
        axis=0,
    )
    np.savetxt(cluster_dir / "multiple_images_param.txt", full_array, fmt="%s", delimiter=",")
    np.savetxt(cluster_dir / "multiple_images_number.txt", np.asarray([counts], dtype=int), fmt="%d", delimiter=",")


def _collect_point_source_catalog(
    *,
    cluster: str,
    path_clumps: str,
    catalog: Dict[str, Any],
    df,
    size: float,
    sizex: Sequence[float],
    sizey: Sequence[float],
    npix: int,
    z_s: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    n_cl = int(catalog["N_cl"])
    if n_cl == 0:
        return [], [], [], []

    workers = min(os.cpu_count() or 1, max(1, n_cl // 6 or 1))
    chunk_size = max(1, int(np.ceil(n_cl / workers)))
    slices = [slice(start, min(start + chunk_size, n_cl)) for start in range(0, n_cl, chunk_size)]

    def _worker(sl: slice) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        point_x: List[np.ndarray] = []
        point_y: List[np.ndarray] = []
        point_mu: List[np.ndarray] = []
        point_mu_tg: List[np.ndarray] = []
        for i in range(sl.start, sl.stop):
            ps = pointsrc(
                size=size,
                sizex=sizex,
                sizey=sizey,
                Npix=npix,
                gl=df,
                ys1=float(catalog["pos_x_arcsec"][i]),
                ys2=float(catalog["pos_y_arcsec"][i]),
                flux=1.0,
                zs=z_s,
            )
            xi, yi, mui = ps.find_images()
            point_x.append(np.asarray(xi, dtype=float))
            point_y.append(np.asarray(yi, dtype=float))
            point_mu.append(np.asarray(mui, dtype=float))
            point_mu_tg.append(np.asarray(mui, dtype=float))
        return point_x, point_y, point_mu, point_mu_tg

    if workers == 1 or len(slices) == 1:
        return _worker(slices[0])

    point_x_all: List[np.ndarray] = []
    point_y_all: List[np.ndarray] = []
    point_mu_all: List[np.ndarray] = []
    point_mu_tg_all: List[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for point_x, point_y, point_mu, point_mu_tg in executor.map(_worker, slices):
            point_x_all.extend(point_x)
            point_y_all.extend(point_y)
            point_mu_all.extend(point_mu)
            point_mu_tg_all.extend(point_mu_tg)
    return point_x_all, point_y_all, point_mu_all, point_mu_tg_all


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clumpy galaxy simulation workflow")
    parser.add_argument(
        "--input-file",
        default=os.environ.get(
            "CLUMPYLEN_INPUT_FILE",
            str(DEFAULT_EXAMPLE_INPUT_FILE if DEFAULT_EXAMPLE_INPUT_FILE.exists() else Path.cwd() / "input" / "input_file.yaml"),
        ),
        help="Path to a sectioned clumPyLen YAML input file",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--rmaxf", type=float, default=8.0, help="Sersic cutoff in effective radii")
    parser.add_argument(
        "--render-scale",
        type=float,
        default=1.0,
        help="Internal rendering scale relative to the final image size. Values below 1 trade fidelity for speed.",
    )
    parser.add_argument(
        "--render-memory-mb",
        type=float,
        default=512.0,
        help="Approximate memory budget used when splitting clump renders into chunks.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    np.random.seed(args.seed)

    cfg = load_input_parameters(args.input_file)
    cluster = str(cfg["cluster"])
    print("We are using the cluster", cluster)

    path_clumps = str(cfg["PATH_TO_clumps"])
    path_fits = str(cfg["PATH_TO_fits"])
    path_host = str(cfg["PATH_TO_host"])
    path_images = str(cfg["PATH_TO_images"])

    for path in [path_clumps, path_fits, path_host, path_images]:
        Path(path).mkdir(parents=True, exist_ok=True)
    for path in [Path(path_clumps) / cluster, Path(path_fits) / cluster, Path(path_host) / cluster, Path(path_images) / cluster]:
        path.mkdir(parents=True, exist_ok=True)

    zl = float(cfg["zl"])
    coords = str(cfg["coords"])
    position = SkyCoord(coords, frame="fk5", unit="deg")
    z_s = float(cfg["z_s"])

    df = create_deflector(
        parfile=str(Path(cfg["PATH_TO_def_angle"]) / f"{cluster}.par"),
        filex=str(Path(cfg["PATH_TO_def_angle"]) / f"{cluster}_angx.fits"),
        filey=str(Path(cfg["PATH_TO_def_angle"]) / f"{cluster}_angy.fits"),
        zl=zl,
        zs=z_s,
        zsnorm=1.0,
        resc_fact=1.0,
    )
    co = df.co
    dl_pc = Distance(z=z_s, unit=u.pc, cosmology=co)

    sizex = cfg["sizex"]
    sizey = cfg["sizey"]
    diffx, diffy = abs(float(sizex[1]) - float(sizex[0])), abs(float(sizey[1]) - float(sizey[0]))
    size = diffx if np.isclose(diffx, diffy) else 50.0
    if not np.isclose(diffx, diffy):
        print("The sizes are different: the image has to be a square... size set to 50 arcsec")

    pix_scale = float(cfg["pix_scale"])
    npix = int(size / pix_scale)
    print("The high resolution images will have", pix_scale, "arcsec/pix")

    gal_x = float(cfg["gal_x"])
    gal_y = float(cfg["gal_y"])
    pix_scale_unl = float(cfg["pix_scale_unl"])
    size_unl = float(cfg["size_unl"])
    npix_unl = int(size_unl / pix_scale_unl)
    render_scale = float(args.render_scale)
    if render_scale <= 0.0:
        raise ValueError("--render-scale must be positive")
    host_render_scale = max(render_scale, MIN_HOST_RENDER_SCALE)
    if host_render_scale != render_scale:
        print(
            f"Host rendering uses a minimum internal scale of {MIN_HOST_RENDER_SCALE:g} "
            f"to preserve lensed morphology; requested render scale {render_scale:g} is still used for fast clump handling."
        )
    grid_cache = _build_render_grid_cache(
        df=df,
        z_s=z_s,
        size=size,
        npix=npix,
        size_unl=size_unl,
        npix_unl=npix_unl,
        sizex=sizex,
        sizey=sizey,
        gal_x=gal_x,
        gal_y=gal_y,
        render_scale=host_render_scale,
    )

    mass_total = float(cfg["mass_total"])
    mass_bul = float(cfg["Mass_bul"])
    mass_disk = float(cfg["Mass_disk"])
    mass_bul_ref = float(cfg["Mass_bul_Y"])
    mass_disk_ref = float(cfg["Mass_disk_Y"])
    fsub = float(cfg["fsub"])
    disk_inclination_deg = float(cfg.get("disk_inclination_deg", 0.0))
    disk_inclination = np.deg2rad(np.clip(disk_inclination_deg, 0.0, 90.0))
    perlin_scale, perlin_octaves, perlin_persistence, perlin_lacunarity, perlin_seed = _perlin_parameters(cfg)
    print(f"Disk inclination requested: {disk_inclination_deg:g} deg")
    print("The clumps mass fraction is", fsub)

    answer_cl = _prompt_yes_no("Do you want to create new clumps?")
    catalog: Dict[str, Any]
    if answer_cl == "yes":
        print("Let us create new clumps!")
        ref_zp = 26.2687
        ob_ref = observation(size=size, Npix=npix, zp=ref_zp)

        print("We start from the host galaxy: building the disk...")
        mag_Y_05Gy_H_WFC3 = np.asarray(cfg["Mag_Y_05Gy_H_WFC3"], dtype=float)
        mag_Y_1Gy_H_WFC3 = np.asarray(cfg["Mag_Y_1Gy_H_WFC3"], dtype=float)
        mass_disk_host = mass_disk * (1.0 - fsub)
        mag_05Gy_disk = mag_Y_05Gy_H_WFC3[0] - (-2.5 * np.log10(mass_disk_ref / mass_disk_host))
        mag_1Gy_bul = mag_Y_1Gy_H_WFC3[0] - (-2.5 * np.log10(mass_bul_ref / mass_bul))
        disk_flux = ob_ref.mag2counts(mag_05Gy_disk)
        bulge_flux = ob_ref.mag2counts(mag_1Gy_bul)

        n_disk = float(cfg["n_disk"])
        re_disk = float(cfg["Re_disk"])
        q_disk = float(cfg["q_disk"])
        if not np.isclose(q_disk, 1.0):
            print("Note: q_disk is ignored for the host disk; clumPyLens keeps the disk circular and uses disk_inclination_deg for the inclination.")
        print("Using a circular disk (q=1.0) with inclination-driven spiral geometry.")
        spiral_A, spiral_alpha, spiral_Phid = _spiral_parameters(cfg, defaults=(5.0, 10.0, 100.0))
        print(f"Spiral parameters: A={spiral_A:g}, alpha={spiral_alpha:g}, Phid={spiral_Phid:g}")
        pa_disk = float(cfg["pa_disk"])
        se_disk, se_disk_unl = _render_single_component_from_cache(
            grid_cache=grid_cache,
            size=size,
            npix=npix,
            size_unl=size_unl,
            npix_unl=npix_unl,
            sizex=sizex,
            sizey=sizey,
            gal_x=gal_x,
            gal_y=gal_y,
            ys1=gal_x,
            ys2=gal_y,
            flux=disk_flux,
            z_s=z_s,
            n=n_disk,
            re=re_disk,
            q=1.0,
            pa=pa_disk,
            rmaxf=args.rmaxf,
            render_memory_mb=args.render_memory_mb,
        )
        se = se_disk.copy()
        se_unl = se_disk_unl.copy()

        if _is_enabled(cfg["ask_spiral"]):
            print("Adding the spiral arms to the disk")
            se, se_unl = _apply_spiral(
                se_disk,
                se_disk_unl,
                grid_cache=grid_cache,
                gal_x=gal_x,
                gal_y=gal_y,
                n=n_disk,
                re=re_disk,
                pa=pa_disk,
                flux=disk_flux,
                A=spiral_A,
                Na=float(cfg["Na"]),
                Phid=spiral_Phid,
                alpha=spiral_alpha,
                phi=disk_inclination,
            )
        else:
            print("Attention: no spiral arms added!")

        if _is_enabled(cfg["ask_Perlin"]):
            print("Adding the Perlin noise to the disk")
            se, se_unl = _apply_perlin(
                se,
                se_unl,
                scale=perlin_scale,
                octaves=perlin_octaves,
                persistence=perlin_persistence,
                lacunarity=perlin_lacunarity,
                seed=perlin_seed,
            )
        else:
            print("Attention: no Perlin noise added!")

        print("We start from the host galaxy: building the spherical bulge...")
        n_bul = float(cfg["n_bul"])
        re_bul = float(cfg["Re_bul"])
        q_bul = float(cfg["q_bul"])
        pa_bul = float(cfg["pa_bul"])
        se_bul, se_bul_unl = _render_single_component_from_cache(
            grid_cache=grid_cache,
            size=size,
            npix=npix,
            size_unl=size_unl,
            npix_unl=npix_unl,
            sizex=sizex,
            sizey=sizey,
            gal_x=gal_x,
            gal_y=gal_y,
            ys1=gal_x,
            ys2=gal_y,
            flux=bulge_flux,
            z_s=z_s,
            n=n_bul,
            re=re_bul,
            q=q_bul,
            pa=pa_bul,
            rmaxf=args.rmaxf,
            render_memory_mb=args.render_memory_mb,
        )

        seGAL = se_bul.copy() + se
        seGAL_unl = se_bul_unl.copy() + se_unl

        alpha = float(cfg["alpha"])
        gamma = float(cfg["gamma"])
        delta = float(cfg["delta"])
        mass_min = float(cfg["mass_min"])
        mass_max = float(cfg["mass_max"])
        trunc_mass = float(cfg["trunc_mass"])
        alpha_t = float(cfg["alpha_t"])
        t_min = float(cfg["t_min"])
        t_max = float(cfg["t_max"])

        print("Let us create the clumps!")
        catalog = _make_clump_catalog(
            npix_unl=npix_unl,
            size_unl=size_unl,
            seGAL_unl=seGAL_unl,
            dl_pc=dl_pc,
            zs=z_s,
            gal_x=gal_x,
            gal_y=gal_y,
            fsub=fsub,
            mass_bulge=mass_bul,
            mass_disk=mass_disk,
            mass_total=mass_total,
            alpha=alpha,
            gamma=gamma,
            delta=delta,
            mass_min=mass_min,
            mass_max=mass_max,
            trunc_mass=trunc_mass,
            alpha_t=alpha_t,
            t_min=t_min,
            t_max=t_max,
            co=co,
        )

        _save_clump_catalog(cluster, path_clumps, catalog)
        _plot_clumps(
            cluster,
            path_images,
            catalog["mass_cl"],
            np.asarray(catalog["reff_pc"], dtype=float),
            np.asarray(catalog["ages"], dtype=float),
        )
    else:
        print("Already created: reading txt file.")
        catalog = _load_clump_catalog(cluster, path_clumps)
        if not np.isclose(disk_inclination_deg, 0.0):
            print("Note: regenerate clumps after changing disk_inclination_deg if you want the catalog positions to follow the new projected disk.")

    print("Do you want to create HR or instrumental observations?")
    answer_instr = _prompt_choice("Select observation type:", ["HR", "Instrument"])

    if answer_instr == "HR":
        print("Let us create HR simulation!")
        print("We start from the host galaxy: building the disk...")

        blue = float(cfg["blue_"])
        green = float(cfg["green_"])
        red = float(cfg["red_"])
        answer_color = _prompt_choice(f"Which color? B = {blue}\\AA, G = {green}\\AA, R = {red}\\AA", ["B", "G", "R"])
        band_wavelength = {"B": blue, "G": green, "R": red}[answer_color]

        disk_flux = _nearest_band_flux_per_mass(
            str(Path(cfg["PATH_TO_sed"]) / "sed_arms.sed"),
            z_s,
            band_wavelength,
            dl_pc.value,
            mass_disk_ref,
        ) * (mass_disk * (1.0 - fsub))

        bulge_flux = _nearest_band_flux_per_mass(
            str(Path(cfg["PATH_TO_sed"]) / "sed_bulge.sed"),
            z_s,
            band_wavelength,
            dl_pc.value,
            mass_bul_ref,
        ) * mass_bul

        n_disk = float(cfg["n_disk"])
        re_disk = float(cfg["Re_disk"])
        q_disk = float(cfg["q_disk"])
        if not np.isclose(q_disk, 1.0):
            print("Note: q_disk is ignored for the host disk; clumPyLens keeps the disk circular and uses disk_inclination_deg for the inclination.")
        print("Using a circular disk (q=1.0) with inclination-driven spiral geometry.")
        spiral_A, spiral_alpha, spiral_Phid = _spiral_parameters(cfg, defaults=(1.0, 10.0, 1.0))
        print(f"Spiral parameters: A={spiral_A:g}, alpha={spiral_alpha:g}, Phid={spiral_Phid:g}")
        pa_disk = float(cfg["pa_disk"])
        se_disk, se_disk_unl = _render_single_component_from_cache(
            grid_cache=grid_cache,
            size=size,
            npix=npix,
            size_unl=size_unl,
            npix_unl=npix_unl,
            sizex=sizex,
            sizey=sizey,
            gal_x=gal_x,
            gal_y=gal_y,
            ys1=gal_x,
            ys2=gal_y,
            flux=disk_flux,
            z_s=z_s,
            n=n_disk,
            re=re_disk,
            q=1.0,
            pa=pa_disk,
            rmaxf=args.rmaxf,
            render_memory_mb=args.render_memory_mb,
        )
        se = se_disk.copy()
        se_unl = se_disk_unl.copy()

        if _is_enabled(cfg["ask_spiral"]):
            print("Adding the spiral arms to the disk")
            se, se_unl = _apply_spiral(
                se_disk,
                se_disk_unl,
                grid_cache=grid_cache,
                gal_x=gal_x,
                gal_y=gal_y,
                n=n_disk,
                re=re_disk,
                pa=pa_disk,
                flux=disk_flux,
                A=spiral_A,
                Na=float(cfg["Na"]),
                Phid=spiral_Phid,
                alpha=spiral_alpha,
                phi=disk_inclination,
            )
        else:
            print("Attention: no spiral arms added!")

        if _is_enabled(cfg["ask_Perlin"]):
            print("Adding the Perlin noise to the disk")
            se, se_unl = _apply_perlin(
                se,
                se_unl,
                scale=perlin_scale,
                octaves=perlin_octaves,
                persistence=perlin_persistence,
                lacunarity=perlin_lacunarity,
                seed=perlin_seed,
            )
        else:
            print("Attention: no Perlin noise added!")

        n_bul = float(cfg["n_bul"])
        re_bul = float(cfg["Re_bul"])
        q_bul = float(cfg["q_bul"])
        pa_bul = float(cfg["pa_bul"])
        se_bul, se_bul_unl = _render_single_component_from_cache(
            grid_cache=grid_cache,
            size=size,
            npix=npix,
            size_unl=size_unl,
            npix_unl=npix_unl,
            sizex=sizex,
            sizey=sizey,
            gal_x=gal_x,
            gal_y=gal_y,
            ys1=gal_x,
            ys2=gal_y,
            flux=bulge_flux,
            z_s=z_s,
            n=n_bul,
            re=re_bul,
            q=q_bul,
            pa=pa_bul,
            rmaxf=args.rmaxf,
            render_memory_mb=args.render_memory_mb,
        )

        seGAL = se_bul.copy() + se
        seGAL_unl = se_bul_unl.copy() + se_unl

        ages_cl = np.asarray(catalog["ages_cl"])
        mass_cl = np.asarray(catalog["mass_cl"], dtype=float)
        mass_cl_ref_map = {
            "1e+06": 1.0e5,
            "2e+06": 2.0e5,
            "3e+06": 3.0e5,
            "4e+06": 3.98e5,
            "5e+06": 4.93e5,
            "6e+06": 5.87e5,
            "7e+06": 6.79e5,
            "8e+06": 7.70e5,
            "9e+06": 8.61e5,
            "1e+07": 9.50e5,
            "2e+07": 8.71e5,
            "3e+07": 8.37e5,
            "4e+07": 8.21e5,
            "5e+07": 8.17e5,
            "6e+07": 8.17e5,
            "7e+07": 8.17e5,
            "8e+07": 8.17e5,
            "9e+07": 8.16e5,
            "1e+08": 8.16e5,
            "2e+08": 8.09e5,
            "3e+08": 8.08e5,
            "4e+08": 8.08e5,
            "5e+08": 8.06e5,
        }

        sed_dir = Path(cfg["PATH_TO_sed"])
        unique_ages, inverse = np.unique(ages_cl, return_inverse=True)
        unique_flux = np.asarray(
            [
                _nearest_band_flux_per_mass(
                    str(sed_dir / f"sed_cl_{age_key}.sed"),
                    z_s,
                    band_wavelength,
                    dl_pc.value,
                    mass_cl_ref_map[age_key],
                )
                for age_key in unique_ages
            ],
            dtype=float,
        )
        flux_cl = unique_flux[inverse] * mass_cl

        print("Let us add the clumps!")
        se_total, se_unl_total = _render_clump_point_sources(
            catalog=catalog,
            fluxes=flux_cl,
            df=df,
            size=size,
            npix=npix,
            sizex=sizex,
            sizey=sizey,
            size_unl=size_unl,
            npix_unl=npix_unl,
            gal_x=gal_x,
            gal_y=gal_y,
            z_s=z_s,
        )

        seGAL += se_total
        seGAL_unl += se_unl_total

        if seGAL.shape != (npix, npix):
            seGAL = _resample_flux_preserving(seGAL, (npix, npix))
        if seGAL_unl.shape != (npix_unl, npix_unl):
            seGAL_unl = _resample_flux_preserving(seGAL_unl, (npix_unl, npix_unl))

        hdu = fits.PrimaryHDU(seGAL)
        _format_header(hdu, position.ra.degree, position.dec.degree, size / ((npix - 1) * 3600.0), npix)
        fits.HDUList([hdu]).writeto(Path(path_fits) / cluster / f"{answer_color}_HR.fits", overwrite=True)

        hdu = fits.PrimaryHDU(seGAL_unl)
        _format_header(hdu, position.ra.degree, position.dec.degree, size_unl / ((npix_unl - 1) * 3600.0), npix_unl)
        fits.HDUList([hdu]).writeto(Path(path_fits) / cluster / f"{answer_color}_HR_unl.fits", overwrite=True)

    elif answer_instr == "Instrument":
        print("Let us create real simulation!")
        print("Let us choose the telescope!")
        answer_tel = _prompt_choice("Select telescope:", TELESCOPE_NAMES)
        filters = FILTERS_BY_TELESCOPE[answer_tel]
        answer_fil = _prompt_choice("Select a filter:", filters)
        filter_idx = filters.index(answer_fil)

        print("The telescope chosen is", answer_tel)
        print("The filter chosen is", answer_fil)
        ntel = TELESCOPE_NAMES.index(answer_tel)
        if answer_tel == "JWST_SW":
            tel_params = {
                "filters": FILTERS_BY_TELESCOPE[answer_tel],
                "ZP": np.asarray(cfg["ZP_J_SW"], dtype=float),
                "Mag_Y_1Gy": np.asarray(cfg["Mag_Y_1Gy_J_SW"], dtype=float),
                "Mag_Y_05Gy": np.asarray(cfg["Mag_Y_05Gy_J_SW"], dtype=float),
                "Texp": np.asarray(cfg["Texp_J_SW"], dtype=float),
                "Flux_sky": np.asarray(cfg["Flux_sky_J_SW"], dtype=float),
                "pix_scale_psf": np.asarray(cfg["pix_scl_PSF_J_SW"], dtype=float),
            }
        elif answer_tel == "JWST_LW":
            tel_params = {
                "filters": FILTERS_BY_TELESCOPE[answer_tel],
                "ZP": np.asarray(cfg["ZP_J_LW"], dtype=float),
                "Mag_Y_1Gy": np.asarray(cfg["Mag_Y_1Gy_J_LW"], dtype=float),
                "Mag_Y_05Gy": np.asarray(cfg["Mag_Y_05Gy_J_LW"], dtype=float),
                "Texp": np.asarray(cfg["Texp_J_LW"], dtype=float),
                "Flux_sky": np.asarray(cfg["Flux_sky_J_LW"], dtype=float),
                "pix_scale_psf": np.asarray(cfg["pix_scl_PSF_J_LW"], dtype=float),
            }
        elif answer_tel == "HST_ACS":
            tel_params = {
                "filters": FILTERS_BY_TELESCOPE[answer_tel],
                "ZP": np.asarray(cfg["ZP_H_ACS"], dtype=float),
                "Mag_Y_1Gy": np.asarray(cfg["Mag_Y_1Gy_H_ACS"], dtype=float),
                "Mag_Y_05Gy": np.asarray(cfg["Mag_Y_05Gy_H_ACS"], dtype=float),
                "Texp": np.asarray(cfg["Texp_H_ACS"], dtype=float),
                "Flux_sky": np.asarray(cfg["Flux_sky_H_ACS"], dtype=float),
                "pix_scale_psf": np.asarray(cfg["pix_scl_PSF_H_ACS"], dtype=float),
            }
        else:
            tel_params = {
                "filters": FILTERS_BY_TELESCOPE[answer_tel],
                "ZP": np.asarray(cfg["ZP_H_WFC3"], dtype=float),
                "Mag_Y_1Gy": np.asarray(cfg["Mag_Y_1Gy_H_WFC3"], dtype=float),
                "Mag_Y_05Gy": np.asarray(cfg["Mag_Y_05Gy_H_WFC3"], dtype=float),
                "Texp": np.asarray(cfg["Texp_H_WFC3"], dtype=float),
                "Flux_sky": np.asarray(cfg["Flux_sky_H_WFC3"], dtype=float),
                "pix_scale_psf": np.asarray(cfg["pix_scl_PSF_H_WFC3"], dtype=float),
            }

        filter_idx = tel_params["filters"].index(answer_fil)
        zp = float(tel_params["ZP"][filter_idx])
        ob = observation(size=size, Npix=npix, zp=zp)

        print("We start from the host galaxy: building the disk...")
        mass_disk_host = mass_disk * (1.0 - fsub)
        mag_Y_05Gy = float(tel_params["Mag_Y_05Gy"][filter_idx])
        delta_mag_disk = -2.5 * np.log10(float(cfg["Mass_disk_Y"]) / mass_disk_host)
        mag_05Gy_disk = mag_Y_05Gy - delta_mag_disk
        _write_key_value(str(Path(path_host) / cluster / "disk_mag.txt"), answer_fil, mag_05Gy_disk)
        fl_disk = ob.mag2counts(mag_05Gy_disk)

        n_disk = float(cfg["n_disk"])
        re_disk = float(cfg["Re_disk"])
        q_disk = float(cfg["q_disk"])
        if not np.isclose(q_disk, 1.0):
            print("Note: q_disk is ignored for the host disk; clumPyLens keeps the disk circular and uses disk_inclination_deg for the inclination.")
        print("Using a circular disk (q=1.0) with inclination-driven spiral geometry.")
        spiral_A, spiral_alpha, spiral_Phid = _spiral_parameters(cfg, defaults=(1.0, 10.0, 1.0))
        print(f"Spiral parameters: A={spiral_A:g}, alpha={spiral_alpha:g}, Phid={spiral_Phid:g}")
        pa_disk = float(cfg["pa_disk"])
        se_disk, se_disk_unl = _render_single_component_from_cache(
            grid_cache=grid_cache,
            size=size,
            npix=npix,
            size_unl=size_unl,
            npix_unl=npix_unl,
            sizex=sizex,
            sizey=sizey,
            gal_x=gal_x,
            gal_y=gal_y,
            ys1=gal_x,
            ys2=gal_y,
            flux=fl_disk,
            z_s=z_s,
            n=n_disk,
            re=re_disk,
            q=1.0,
            pa=pa_disk,
            rmaxf=args.rmaxf,
            render_memory_mb=args.render_memory_mb,
        )
        se = se_disk.copy()
        se_unl = se_disk_unl.copy()

        if _is_enabled(cfg["ask_spiral"]):
            print("Adding the spiral arms to the disk")
            se, se_unl = _apply_spiral(
                se_disk,
                se_disk_unl,
                grid_cache=grid_cache,
                gal_x=gal_x,
                gal_y=gal_y,
                n=n_disk,
                re=re_disk,
                pa=pa_disk,
                flux=fl_disk,
                A=spiral_A,
                Na=float(cfg["Na"]),
                Phid=spiral_Phid,
                alpha=spiral_alpha,
                phi=disk_inclination,
            )
        else:
            print("Attention: no spiral arms added!")

        if _is_enabled(cfg["ask_Perlin"]):
            print("Adding the Perlin noise to the disk")
            se, se_unl = _apply_perlin(
                se,
                se_unl,
                scale=perlin_scale,
                octaves=perlin_octaves,
                persistence=perlin_persistence,
                lacunarity=perlin_lacunarity,
                seed=perlin_seed,
            )
        else:
            print("Attention: no Perlin noise added!")

        print("We start from the host galaxy: building the spherical bulge...")
        n_bul = float(cfg["n_bul"])
        re_bul = float(cfg["Re_bul"])
        q_bul = float(cfg["q_bul"])
        pa_bul = float(cfg["pa_bul"])
        mag_Y_1Gy = float(tel_params["Mag_Y_1Gy"][filter_idx])
        delta_mag_bul = -2.5 * np.log10(float(cfg["Mass_bul_Y"]) / mass_bul)
        mag_1Gy_bul = mag_Y_1Gy - delta_mag_bul
        _write_key_value(str(Path(path_host) / cluster / "bulge_mag.txt"), answer_fil, mag_1Gy_bul)
        fl_bul = ob.mag2counts(mag_1Gy_bul)

        se_bul, se_bul_unl = _render_single_component_from_cache(
            grid_cache=grid_cache,
            size=size,
            npix=npix,
            size_unl=size_unl,
            npix_unl=npix_unl,
            sizex=sizex,
            sizey=sizey,
            gal_x=gal_x,
            gal_y=gal_y,
            ys1=gal_x,
            ys2=gal_y,
            flux=fl_bul,
            z_s=z_s,
            n=n_bul,
            re=re_bul,
            q=q_bul,
            pa=pa_bul,
            rmaxf=args.rmaxf,
            render_memory_mb=args.render_memory_mb,
        )

        seGAL = se_bul + se
        seGAL_unl = se_bul_unl + se_unl

        print("Building the clumps...")
        ages_cl = np.asarray(catalog["ages_cl"])
        mass_cl = np.asarray(catalog["mass_cl"], dtype=float)
        mag_cl = _instrument_mag_scalars(
            cfg=cfg,
            cluster=cluster,
            telescope_name=answer_tel,
            filter_idx=filter_idx,
            ages_cl=ages_cl,
            mass_cl=mass_cl,
        )
        flux_sed = ob.mag2counts(mag_cl)
        _write_table(
            str(Path(path_clumps) / cluster / "clumps_mag_column.txt"),
            ["ID", "Stellar_Mass", "Age_yr", answer_fil],
            [
                np.arange(catalog["N_cl"], dtype=int),
                mass_cl,
                np.asarray(catalog["ages"], dtype=float),
                mag_cl,
            ],
        )

        se_total, se_unl_total = _render_clump_point_sources(
            catalog=catalog,
            fluxes=flux_sed,
            df=df,
            size=size,
            npix=npix,
            sizex=sizex,
            sizey=sizey,
            size_unl=size_unl,
            npix_unl=npix_unl,
            gal_x=gal_x,
            gal_y=gal_y,
            z_s=z_s,
        )

        seGAL += se_total
        seGAL_unl += se_unl_total

        if seGAL.shape != (npix, npix):
            seGAL = _resample_flux_preserving(seGAL, (npix, npix))
        if seGAL_unl.shape != (npix_unl, npix_unl):
            seGAL_unl = _resample_flux_preserving(seGAL_unl, (npix_unl, npix_unl))

        hdu = fits.PrimaryHDU(seGAL)
        _format_header(hdu, position.ra.degree, position.dec.degree, size / ((npix - 1) * 3600.0), npix)
        fits.HDUList([hdu]).writeto(Path(path_fits) / cluster / f"{answer_fil}_{answer_tel}_HR.fits", overwrite=True)

        hdu = fits.PrimaryHDU(seGAL_unl)
        _format_header(hdu, position.ra.degree, position.dec.degree, size_unl / ((npix_unl - 1) * 3600.0), npix_unl)
        fits.HDUList([hdu]).writeto(Path(path_fits) / cluster / f"{answer_fil}_{answer_tel}_HR_unl.fits", overwrite=True)

        pix_scale_ins = float(np.asarray(cfg["pix_scl_instr"], dtype=float)[ntel])
        npix_ins = int(size / pix_scale_ins)
        npix_ins_unl = int(size_unl / pix_scale_ins)
        texp = float(tel_params["Texp"][filter_idx])
        flux_sky = float(tel_params["Flux_sky"][filter_idx])

        ob_inst = observation(size=size, Npix=npix_ins, zp=zp, texp=texp, bkg=flux_sky, bkg_counts_in=True)
        ob_inst_unl = observation(size=size_unl, Npix=npix_ins_unl, zp=zp, texp=texp, bkg=flux_sky, bkg_counts_in=True)

        psf_path = Path(cfg["PATH_TO_psf"]) / f"{answer_tel}_PSF_{answer_fil}.fits"
        with fits.open(psf_path) as hdul:
            psf_true = hdul[0].data
        psf_true_lens = resize_psf(psf_true, float(tel_params["pix_scale_psf"][filter_idx]), pix_scale, order=1)
        psf_true_lens /= np.sum(psf_true_lens)
        with_psf = fftconvolve(seGAL, psf_true_lens, mode="same")

        psf_true_unlens = resize_psf(psf_true, float(tel_params["pix_scale_psf"][filter_idx]), pix_scale_unl, order=1)
        psf_true_unlens /= np.sum(psf_true_unlens)
        with_psf_unl = fftconvolve(seGAL_unl, psf_true_unlens, mode="same")

        hdu = fits.PrimaryHDU(with_psf)
        _format_header(hdu, position.ra.degree, position.dec.degree, size / ((npix - 1) * 3600.0), npix)
        fits.HDUList([hdu]).writeto(Path(path_fits) / cluster / f"{answer_fil}_{answer_tel}_psf.fits", overwrite=True)

        with_psf_telescope = zoom(with_psf, npix_ins / npix, order=1)
        with_psf_telescope_unl = zoom(with_psf_unl, npix_ins_unl / npix_unl, order=1)

        flux_tot = float(with_psf.sum())
        flux_tot_unl = float(with_psf_unl.sum())
        if np.sum(with_psf_telescope) > 0:
            with_psf_telescope *= flux_tot / float(np.sum(with_psf_telescope))
        if np.sum(with_psf_telescope_unl) > 0:
            with_psf_telescope_unl *= flux_tot_unl / float(np.sum(with_psf_telescope_unl))

        with_psf_sky = with_psf_telescope + flux_sky
        with_psf_sky_unl = with_psf_telescope_unl + flux_sky

        toshow = with_psf_sky + ob_inst.makeNoise(with_psf_sky)
        toshow_unl = with_psf_sky_unl + ob_inst_unl.makeNoise(with_psf_sky_unl)

        hdu = fits.PrimaryHDU(toshow)
        _format_header(hdu, position.ra.degree, position.dec.degree, size / ((npix_ins - 1) * 3600.0), npix_ins)
        fits.HDUList([hdu]).writeto(Path(path_fits) / cluster / f"{answer_fil}_{answer_tel}.fits", overwrite=True)

        hdu = fits.PrimaryHDU(toshow_unl)
        _format_header(hdu, position.ra.degree, position.dec.degree, size_unl / ((npix_ins_unl - 1) * 3600.0), npix_ins_unl)
        fits.HDUList([hdu]).writeto(Path(path_fits) / cluster / f"{answer_fil}_{answer_tel}_unlensed.fits", overwrite=True)

    else:
        raise ValueError(f"Unknown observation mode: {answer_instr}")


if __name__ == "__main__":
    main()
