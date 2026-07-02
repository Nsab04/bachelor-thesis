#!/usr/bin/env python3
"""
Compare reference and reconstructed deflection maps with pyLensLib's deflector.

The script loads deflection-angle maps stored as NumPy arrays, builds a
map-based deflector for each case, computes the convergence map, and overlays
the tangential/radial critical lines for a list of source redshifts.
It also computes the lensing potential maps, compares them through a relative
difference map, and shows the pixel-value distribution.

An optional time-delay example mode can sample random sources inside the
caustics and visualize the corresponding time-delay surfaces and contours.

Example
-------
python apps/xincheng/plot_deflection_outputs.py \
    --zl 0.3 \
    --pixel-scale-arcsec 0.5 \
    --input-dir /Users/maxmen3/Downloads/deflection_outputs
"""

from __future__ import annotations

import argparse
import sys
import string
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import map_coordinates
from shapely.geometry import Point
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyLensLib.deflector import deflector
from pyLensLib.pointsrc import pointsrc
from pyLensLib.sersic import sersic


DEFAULT_INPUT_DIR = Path("/Users/maxmen3/Downloads/deflection_outputs")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "deflection_ref_recon_critical_lines.png"
DEFAULT_POTENTIAL_OUTPUT = DEFAULT_INPUT_DIR / "deflection_potential_relative_difference.png"
DEFAULT_TD_CAUSTICS_OUTPUT = DEFAULT_INPUT_DIR / "deflection_time_delay_caustics_examples.png"
DEFAULT_TD_OUTPUT = DEFAULT_INPUT_DIR / "deflection_time_delay_surfaces_examples.png"
DEFAULT_TD_IMAGES_OUTPUT = DEFAULT_INPUT_DIR / "deflection_time_delay_sersic_images_examples.png"
DEFAULT_ZS = (1.0, 3.0, 6.0)


def resolve_output_path(output_path: Path, default_filename: str) -> Path:
    output_path = output_path.expanduser().resolve()
    if (output_path.exists() and output_path.is_dir()) or output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / default_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display reference/reconstructed convergence maps with critical lines."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing ref_/recon_ deflection .npy files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output figure path or directory. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--potential-output",
        type=Path,
        default=DEFAULT_POTENTIAL_OUTPUT,
        help=f"Output path or directory for the potential-difference figure. Default: {DEFAULT_POTENTIAL_OUTPUT}",
    )
    parser.add_argument(
        "--zl",
        type=float,
        required=True,
        help="Lens redshift used to build the deflector.",
    )
    parser.add_argument(
        "--pixel-scale-arcsec",
        type=float,
        required=True,
        help="Pixel scale of the deflection maps in arcsec/pixel.",
    )
    parser.add_argument(
        "--zs-norm",
        type=float,
        default=1.6,
        help="Source redshift assumed when the maps were generated. Default: 1.6",
    )
    parser.add_argument(
        "--h0",
        type=float,
        default=70.0,
        help="Hubble constant for the plotting cosmology. Default: 70.0",
    )
    parser.add_argument(
        "--om0",
        type=float,
        default=0.3,
        help="Matter density for the plotting cosmology. Default: 0.3",
    )
    parser.add_argument(
        "--zs",
        type=float,
        nargs="+",
        default=list(DEFAULT_ZS),
        help="Source redshifts to plot. Default: 1 3 6",
    )
    parser.add_argument(
        "--potential-mask-radius",
        type=float,
        default=60.0,
        help="Radius in arcsec to mask out from the potential-difference map. Default: 60.0",
    )
    parser.add_argument(
        "--potential-hist-bins",
        type=int,
        default=60,
        help="Number of histogram bins for the potential-difference distribution. Default: 60",
    )
    parser.add_argument(
        "--td-examples",
        "--show-td-examples",
        dest="show_td_examples",
        action="store_true",
        help="Generate random source / time-delay example panels for REF and RECON.",
    )
    parser.add_argument(
        "--td-output",
        type=Path,
        default=DEFAULT_TD_OUTPUT,
        help=f"Output path or directory for the time-delay surface figure. Default: {DEFAULT_TD_OUTPUT}",
    )
    parser.add_argument(
        "--td-images-output",
        type=Path,
        default=DEFAULT_TD_IMAGES_OUTPUT,
        help=f"Output path or directory for the lensed Sérsic image figure. Default: {DEFAULT_TD_IMAGES_OUTPUT}",
    )
    parser.add_argument(
        "--td-sersic-re",
        type=float,
        default=1.0,
        help="Effective radius in arcsec for the Sérsic source used in the lensed image panels. Default: 1.0",
    )
    parser.add_argument(
        "--td-caustics-output",
        type=Path,
        default=DEFAULT_TD_CAUSTICS_OUTPUT,
        help=f"Output path or directory for the caustic/source-plane example figure. Default: {DEFAULT_TD_CAUSTICS_OUTPUT}",
    )
    parser.add_argument(
        "--td-n-rad",
        type=int,
        default=1,
        help="Number of examples to place inside the radial caustic. Default: 1",
    )
    parser.add_argument(
        "--td-n-tan",
        type=int,
        default=2,
        help="Number of examples to place inside the tangential caustic. Default: 2",
    )
    parser.add_argument(
        "--td-zmin",
        type=float,
        default=1.0,
        help="Lower bound for source redshifts in the time-delay example mode. Default: 1.0",
    )
    parser.add_argument(
        "--td-zmax",
        type=float,
        default=3.0,
        help="Upper bound for source redshifts in the time-delay example mode. Default: 3.0",
    )
    parser.add_argument(
        "--td-seed",
        type=int,
        default=12345,
        help="Random seed for the time-delay example source placement. Default: 12345",
    )
    parser.add_argument(
        "--show-im-diagnostic",
        action="store_true",
        default=False,
        help="Print TD gradient diagnostics inside the image panels. Default: False",
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    show_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Only save the figure.",
    )
    parser.set_defaults(show=True)
    return parser.parse_args()


def load_case(input_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    ax_path = input_dir / f"{prefix}_alpha_x_arcsec.npy"
    ay_path = input_dir / f"{prefix}_alpha_y_arcsec.npy"
    if not ax_path.exists():
        raise FileNotFoundError(f"Missing file: {ax_path}")
    if not ay_path.exists():
        raise FileNotFoundError(f"Missing file: {ay_path}")
    angx = np.load(ax_path)
    angy = np.load(ay_path)
    if angx.shape != angy.shape:
        raise ValueError(f"Shape mismatch for {prefix}: {angx.shape} vs {angy.shape}")
    return angx, angy


def build_theta_grid(npix: int, pixel_scale_arcsec: float) -> np.ndarray:
    center = 0.5 * (npix - 1)
    return (np.arange(npix, dtype=float) - center) * pixel_scale_arcsec


def build_deflector_for_case(
    co: FlatLambdaCDM,
    angx: np.ndarray,
    angy: np.ndarray,
    zl: float,
    zs_norm: float,
    pixel_scale_arcsec: float,
    compute_potential: bool = False,
) -> deflector:
    df = deflector(co, angx=angx, angy=angy, zl=zl, zs=zs_norm)
    theta = build_theta_grid(angx.shape[0], pixel_scale_arcsec)
    df.setGrid(theta=theta, compute_potential=compute_potential)
    return df


def finite_percentile(values: Sequence[np.ndarray], q: float) -> float:
    stacked = np.concatenate([np.ravel(np.asarray(v, dtype=float)) for v in values])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return 1.0
    out = float(np.nanpercentile(finite, q))
    if not np.isfinite(out) or out <= 0.0:
        return 1.0
    return out


def image_label(index: int) -> str:
    """
    Return an alphabetical image label: A, B, ..., Z, AA, AB, ...
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    letters = string.ascii_uppercase
    label = ""
    n = index
    while True:
        n, rem = divmod(n, 26)
        label = letters[rem] + label
        if n == 0:
            break
        n -= 1
    return label


def format_time_delay_summary(labels: list[str], td_values_days: np.ndarray) -> str:
    """
    Format pairwise time delays relative to image A.
    """
    if len(labels) == 0:
        return "no images"
    if len(labels) == 1:
        return f"{labels[0]} = 0 d"
    lines = []
    t0 = float(td_values_days[0])
    for lab, tval in zip(labels[1:], td_values_days[1:]):
        lines.append(f"Δt_{labels[0]}{lab} = {float(tval - t0):.1f} d")
    return "\n".join(lines)


def sample_point_in_geometry(
    geometry,
    rng: np.random.Generator,
    max_tries: int = 10000,
    fallback: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """
    Draw a random point inside a shapely geometry using rejection sampling.
    """
    if geometry.is_empty:
        return float(fallback[0]), float(fallback[1])
    minx, miny, maxx, maxy = geometry.bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        return float(fallback[0]), float(fallback[1])
    for _ in range(max_tries):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if geometry.covers(Point(x, y)):
            return x, y
    rp = geometry.representative_point()
    return float(rp.x), float(rp.y)


def sample_source_inside_caustics(
    df: deflector,
    rng: np.random.Generator,
    min_images: int = 2,
    max_tries: int = 200,
) -> tuple[tuple[float, float], int, bool]:
    """
    Sample a source inside the caustics and keep only draws that produce multiple images.

    The caustic union returned by the lens code is expressed in source-plane pixel units,
    so the sampled point is converted back to arcsec before being passed to the point-source
    solver.
    """
    caustic_union = df.causticsUnaryUnion()
    pixel_fallback = ((df.nray1 - 1) / 2.0, (df.nray2 - 1) / 2.0)

    if caustic_union.is_empty:
        beta = (0.0, 0.0)
        ps = pointsrc(
            size=(df.thetax[-1] - df.thetax[0]),
            Npix=df.nray1,
            gl=df,
            ys1=beta[0],
            ys2=beta[1],
            zs=df.zs,
            refine=True,
        )
        return beta, int(len(ps.xi1)), True

    for _ in range(max_tries):
        beta_pix = sample_point_in_geometry(caustic_union, rng, fallback=pixel_fallback)
        beta = (
            beta_pix[0] * df.pixel_scale + df.thetax[0],
            beta_pix[1] * df.pixel_scale + df.thetay[0],
        )
        ps = pointsrc(
            size=(df.thetax[-1] - df.thetax[0]),
            Npix=df.nray1,
            gl=df,
            ys1=beta[0],
            ys2=beta[1],
            zs=df.zs,
            refine=True,
        )
        if len(ps.xi1) >= min_images:
            return beta, int(len(ps.xi1)), False

    beta_pix = caustic_union.representative_point()
    beta = (
        float(beta_pix.x) * df.pixel_scale + df.thetax[0],
        float(beta_pix.y) * df.pixel_scale + df.thetay[0],
    )
    ps = pointsrc(
        size=(df.thetax[-1] - df.thetax[0]),
        Npix=df.nray1,
        gl=df,
        ys1=beta[0],
        ys2=beta[1],
        zs=df.zs,
        refine=True,
    )
    return beta, int(len(ps.xi1)), True


def sample_common_source_inside_caustics(
    df_ref: deflector,
    df_recon: deflector,
    rng: np.random.Generator,
    min_images: int = 2,
    max_tries: int = 500,
) -> tuple[tuple[float, float], dict[str, int], bool]:
    """
    Sample one source position that is shared by REF and RECON.

    The preferred draw comes from the intersection of the two caustic unions, which keeps
    the source inside both multiply-imaged regions. If that fails, fall back to testing draws
    from either model until both produce at least ``min_images`` images.
    """
    ref_union = df_ref.causticsUnaryUnion()
    recon_union = df_recon.causticsUnaryUnion()
    common_union = ref_union.intersection(recon_union)
    pixel_fallback = ((df_ref.nray1 - 1) / 2.0, (df_ref.nray2 - 1) / 2.0)

    def evaluate(beta_pix: tuple[float, float]) -> tuple[tuple[float, float], int, int]:
        beta = (
            beta_pix[0] * df_ref.pixel_scale + df_ref.thetax[0],
            beta_pix[1] * df_ref.pixel_scale + df_ref.thetay[0],
        )
        ps_ref = pointsrc(
            size=(df_ref.thetax[-1] - df_ref.thetax[0]),
            Npix=df_ref.nray1,
            gl=df_ref,
            ys1=beta[0],
            ys2=beta[1],
            zs=df_ref.zs,
            refine=True,
        )
        ps_recon = pointsrc(
            size=(df_recon.thetax[-1] - df_recon.thetax[0]),
            Npix=df_recon.nray1,
            gl=df_recon,
            ys1=beta[0],
            ys2=beta[1],
            zs=df_recon.zs,
            refine=True,
        )
        return beta, int(len(ps_ref.xi1)), int(len(ps_recon.xi1))

    candidate_areas = []
    if not common_union.is_empty:
        candidate_areas.append(common_union)
    if not ref_union.is_empty:
        candidate_areas.append(ref_union)
    if not recon_union.is_empty:
        candidate_areas.append(recon_union)

    for geom in candidate_areas:
        for _ in range(max_tries):
            beta_pix = sample_point_in_geometry(geom, rng, fallback=pixel_fallback)
            beta, n_ref, n_recon = evaluate(beta_pix)
            if n_ref >= min_images and n_recon >= min_images:
                return beta, {"ref": n_ref, "recon": n_recon}, False

    if not common_union.is_empty:
        beta_pix = common_union.representative_point()
    elif not ref_union.is_empty:
        beta_pix = ref_union.representative_point()
    elif not recon_union.is_empty:
        beta_pix = recon_union.representative_point()
    else:
        beta_pix = Point(pixel_fallback[0], pixel_fallback[1])
    beta, n_ref, n_recon = evaluate((float(beta_pix.x), float(beta_pix.y)))
    return beta, {"ref": n_ref, "recon": n_recon}, True


def build_time_delay_example_schedule(
    td_n_rad: int,
    td_n_tan: int,
    td_zmin: float,
    td_zmax: float,
) -> list[tuple[str, float]]:
    """
    Build a list of (caustic_kind, source_redshift) pairs for the time-delay examples.
    """
    if td_n_rad < 0 or td_n_tan < 0:
        raise ValueError("td_n_rad and td_n_tan must be non-negative.")
    total = td_n_rad + td_n_tan
    if total < 1:
        raise ValueError("At least one time-delay example is required.")
    z_values = np.linspace(float(td_zmin), float(td_zmax), total)
    schedule: list[tuple[str, float]] = [("rad", float(z)) for z in z_values[:td_n_rad]]
    schedule.extend(("tan", float(z)) for z in z_values[td_n_rad:])
    return schedule


def caustic_union_by_kind(df: deflector, caustic_kind: str) -> object:
    """
    Return the unary union for the tangential or radial caustics.
    """
    kind = caustic_kind.lower()
    if kind not in {"rad", "tan"}:
        raise ValueError(f"Unknown caustic kind: {caustic_kind}")
    cl = df.radcl() if kind == "rad" else df.tancl()
    caustics = df.getCaustics(cl)
    geoms = [c.geometria for c in caustics if not c.geometria.is_empty]
    if len(geoms) == 0:
        return unary_union([])
    return unary_union(geoms)


def sample_common_source_inside_caustic_kind(
    df_ref: deflector,
    df_recon: deflector,
    caustic_kind: str,
    rng: np.random.Generator,
    min_images: int = 2,
    max_tries: int = 500,
) -> tuple[tuple[float, float], dict[str, int], bool]:
    """
    Sample one source position shared by REF and RECON inside a chosen caustic family.
    """
    ref_union = caustic_union_by_kind(df_ref, caustic_kind)
    recon_union = caustic_union_by_kind(df_recon, caustic_kind)
    common_union = ref_union.intersection(recon_union)
    pixel_fallback = ((df_ref.nray1 - 1) / 2.0, (df_ref.nray2 - 1) / 2.0)

    def evaluate(beta_pix: tuple[float, float]) -> tuple[tuple[float, float], int, int]:
        beta = (
            beta_pix[0] * df_ref.pixel_scale + df_ref.thetax[0],
            beta_pix[1] * df_ref.pixel_scale + df_ref.thetay[0],
        )
        ps_ref = pointsrc(
            size=(df_ref.thetax[-1] - df_ref.thetax[0]),
            Npix=df_ref.nray1,
            gl=df_ref,
            ys1=beta[0],
            ys2=beta[1],
            zs=df_ref.zs,
            refine=True,
        )
        ps_recon = pointsrc(
            size=(df_recon.thetax[-1] - df_recon.thetax[0]),
            Npix=df_recon.nray1,
            gl=df_recon,
            ys1=beta[0],
            ys2=beta[1],
            zs=df_recon.zs,
            refine=True,
        )
        return beta, int(len(ps_ref.xi1)), int(len(ps_recon.xi1))

    candidate_areas = []
    if not common_union.is_empty:
        candidate_areas.append(common_union)
    if not ref_union.is_empty:
        candidate_areas.append(ref_union)
    if not recon_union.is_empty:
        candidate_areas.append(recon_union)

    for geom in candidate_areas:
        for _ in range(max_tries):
            beta_pix = sample_point_in_geometry(geom, rng, fallback=pixel_fallback)
            beta, n_ref, n_recon = evaluate(beta_pix)
            if n_ref >= min_images and n_recon >= min_images:
                return beta, {"ref": n_ref, "recon": n_recon}, False

    if not common_union.is_empty:
        beta_pix = common_union.representative_point()
    elif not ref_union.is_empty:
        beta_pix = ref_union.representative_point()
    elif not recon_union.is_empty:
        beta_pix = recon_union.representative_point()
    else:
        beta_pix = Point(pixel_fallback[0], pixel_fallback[1])
    beta, n_ref, n_recon = evaluate((float(beta_pix.x), float(beta_pix.y)))
    return beta, {"ref": n_ref, "recon": n_recon}, True


def get_td_contour_levels(td_rel: np.ndarray, image_positions: tuple[np.ndarray, np.ndarray], df: deflector) -> np.ndarray:
    """
    Build contour levels anchored on the time-delay values at image positions.
    """
    xi, yi = image_positions
    if xi.size == 0:
        return np.linspace(float(np.nanmin(td_rel)), float(np.nanmax(td_rel)), 12)
    xpix = xi / df.pixel_scale + (df.nray1 - 1) / 2.0
    ypix = yi / df.pixel_scale + (df.nray2 - 1) / 2.0
    td_img = map_coordinates(td_rel, [ypix, xpix], order=1, mode="nearest")
    td_img = np.asarray(td_img, dtype=float)
    td_img = td_img[np.isfinite(td_img)]
    if td_img.size == 0:
        return np.linspace(float(np.nanmin(td_rel)), float(np.nanmax(td_rel)), 12)
    levels = np.unique(np.round(td_img, 6))
    if levels.size < 2:
        span = float(np.nanmax(td_rel) - np.nanmin(td_rel))
        if not np.isfinite(span) or span <= 0.0:
            span = 1.0
        base = float(levels[0]) if levels.size == 1 else float(np.nanmedian(td_rel))
        levels = np.array([base - 0.25 * span, base, base + 0.25 * span], dtype=float)
    return np.sort(levels)


def plot_caustic_panel(ax: plt.Axes, df: deflector, beta: tuple[float, float], title: str) -> None:
    """
    Plot the source plane, caustics, and the source position.
    """
    tancl = df.tancl()
    radcl = df.radcl()
    tancau = df.getCaustics(tancl)
    radcau = df.getCaustics(radcl)

    bounds = []
    for ca in tancau:
        x, y = df.getCausticPoints(ca)
        ax.plot(x, y, color="tab:blue", lw=1.3, alpha=0.9)
        bounds.append((float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))))
    for ca in radcau:
        x, y = df.getCausticPoints(ca)
        ax.plot(x, y, color="tab:orange", lw=1.1, alpha=0.9)
        bounds.append((float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))))

    ax.plot(beta[0], beta[1], marker="*", color="red", markersize=13, markeredgecolor="black", markeredgewidth=0.8)
    ax.set_title(title, fontsize=9, pad=2)
    ax.set_xlabel(r"$\beta_1$ [arcsec]")
    ax.set_ylabel(r"$\beta_2$ [arcsec]")
    ax.set_aspect("equal")
    if bounds:
        minx = min(b[0] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxx = max(b[2] for b in bounds)
        maxy = max(b[3] for b in bounds)
        pad_x = max(0.05 * (maxx - minx), 0.5)
        pad_y = max(0.05 * (maxy - miny), 0.5)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)


def plot_time_delay_panel(ax: plt.Axes, df: deflector, beta: tuple[float, float], title: str) -> dict[str, float]:
    """
    Plot the time-delay surface, stationary-point contours, and image positions.
    """
    td = df.t_delay_surf(beta=beta)
    td_rel = td - np.nanmin(td)
    grad_y, grad_x = np.gradient(td_rel, df.pixel_scale, df.pixel_scale)
    ps = pointsrc(
        size=(df.thetax[-1] - df.thetax[0]),
        Npix=df.nray1,
        gl=df,
        ys1=beta[0],
        ys2=beta[1],
        zs=df.zs,
        refine=True,
        refine_to_td=True,
        td_surface=td_rel,
        td_grad_x=grad_x,
        td_grad_y=grad_y,
    )
    xi = np.asarray(ps.xi1, dtype=float)
    yi = np.asarray(ps.xi2, dtype=float)
    mui = np.asarray(ps.mui, dtype=float)
    shifts = np.asarray(getattr(ps, "td_refine_shift", np.zeros_like(xi)), dtype=float)
    success = np.asarray(getattr(ps, "td_refine_success", np.ones_like(xi, dtype=bool)), dtype=bool)
    contour_levels = get_td_contour_levels(td_rel, (xi, yi), df)

    vmax = float(np.nanpercentile(td_rel, 99.0)) if np.any(np.isfinite(td_rel)) else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    levels_filled = np.linspace(0.0, vmax, 28)

    extent = [float(df.thetax[0]), float(df.thetax[-1]), float(df.thetay[0]), float(df.thetay[-1])]
    xx, yy = np.meshgrid(df.thetax, df.thetay)
    ax.contourf(xx, yy, td_rel, levels=levels_filled, cmap=cm.coolwarm, alpha=0.95)
    ax.contour(xx, yy, td_rel, levels=contour_levels, colors="white", linewidths=2.0)
    ax.contour(xx, yy, td_rel, levels=contour_levels, colors="black", linewidths=0.8, alpha=0.8)

    tancl = df.tancl()
    radcl = df.radcl()
    for cl in tancl:
        x, y = df.getCritPoints(cl)
        ax.plot(x, y, color="orange", lw=0.8, alpha=0.95, ls=":")
    for cl in radcl:
        x, y = df.getCritPoints(cl)
        ax.plot(x, y, color="orange", lw=0.8, alpha=0.95, ls=":")

    xpix = xi / df.pixel_scale + (df.nray1 - 1) / 2.0
    ypix = yi / df.pixel_scale + (df.nray2 - 1) / 2.0
    td_img = map_coordinates(td_rel, [ypix, xpix], order=1, mode="nearest")
    td_img = np.asarray(td_img, dtype=float)
    finite = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(td_img)
    xi = xi[finite]
    yi = yi[finite]
    td_img = td_img[finite]
    shifts = shifts[finite]
    success = success[finite]
    order = np.argsort(td_img)
    xi = xi[order]
    yi = yi[order]
    td_img = td_img[order]
    shifts = shifts[order]
    success = success[order]
    labels = [image_label(i) for i in range(len(xi))]

    grad_x_img = map_coordinates(grad_x, [ypix, xpix], order=1, mode="nearest")
    grad_y_img = map_coordinates(grad_y, [ypix, xpix], order=1, mode="nearest")
    grad_x_img = np.asarray(grad_x_img, dtype=float)[finite][order]
    grad_y_img = np.asarray(grad_y_img, dtype=float)[finite][order]
    grad_norm = np.sqrt(grad_x_img**2 + grad_y_img**2)
    good = np.isfinite(grad_norm) & success
    if np.any(good):
        grad_norm = grad_norm[good]
    else:
        grad_norm = grad_norm[np.isfinite(grad_norm)]
    max_grad = float(np.nanmax(grad_norm)) if grad_norm.size else float("nan")
    med_grad = float(np.nanmedian(grad_norm)) if grad_norm.size else float("nan")
    max_shift = float(np.nanmax(shifts)) if shifts.size else float("nan")
    med_shift = float(np.nanmedian(shifts)) if shifts.size else float("nan")
    n_failed = int(np.count_nonzero(~success))

    ax.plot(xi, yi, linestyle="None", marker="*", markersize=12, color="yellow", markeredgecolor="black", markeredgewidth=0.8)
    span_x = float(np.nanmax(xx) - np.nanmin(xx)) if np.any(np.isfinite(xx)) else 1.0
    span_y = float(np.nanmax(yy) - np.nanmin(yy)) if np.any(np.isfinite(yy)) else 1.0
    dx = 0.02 * span_x if span_x > 0 else 0.1
    dy = 0.02 * span_y if span_y > 0 else 0.1
    for x, y, lab in zip(xi, yi, labels):
        ax.text(
            x + dx,
            y + dy,
            lab,
            fontsize=10,
            weight="bold",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.12),
            zorder=6,
        )
    ax.set_title(title, fontsize=9, pad=2)
    ax.set_xlabel(r"$\theta_1$ [arcsec]")
    ax.set_ylabel(r"$\theta_2$ [arcsec]")
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    summary = format_time_delay_summary(labels, td_img)
    ax.text(
        0.03,
        0.97,
        f"images={len(xi)}\n"
        f"z={df.zs:.2f}\n"
        f"{summary}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.25),
    )
    return {
        "max_grad": max_grad,
        "median_grad": med_grad,
        "max_shift": max_shift,
        "median_shift": med_shift,
        "n_failed": n_failed,
    }


def build_lensed_sersic_image(
    df: deflector,
    beta: tuple[float, float],
    stamp_size_arcsec: float = 120.0,
    source_re_arcsec: float = 1.0,
) -> np.ndarray:
    """
    Render a lensed circular Sersic source for the given lens and source position.
    """
    stamp_npix = max(3, int(round(stamp_size_arcsec / float(df.pixel_scale))) + 1)
    se = sersic(
        size=float(stamp_size_arcsec),
        Npix=stamp_npix,
        gl=df,
        save_unlensed=False,
        zs=float(df.zs),
        ys1=float(beta[0]),
        ys2=float(beta[1]),
        q=1.0,
        pa=0.0,
        re=float(source_re_arcsec),
        n=1.0,
        flux=1.0,
    )
    return np.asarray(se.image, dtype=float)


def plot_lensed_sersic_panel(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    stamp_size_arcsec: float,
    norm: LogNorm,
) -> None:
    """
    Plot a lensed Sersic postage stamp.
    """
    half_size = 0.5 * float(stamp_size_arcsec)
    ax.imshow(
        image,
        origin="lower",
        extent=[-half_size, half_size, -half_size, half_size],
        cmap="magma",
        norm=norm,
    )
    ax.axhline(0.0, color="white", lw=0.6, alpha=0.35)
    ax.axvline(0.0, color="white", lw=0.6, alpha=0.35)
    ax.set_title(title, fontsize=9, pad=2)
    ax.set_xlabel(r"$\theta_1$ [arcsec]")
    ax.set_ylabel(r"$\theta_2$ [arcsec]")
    ax.set_aspect("equal")


def plot_lensed_sersic_residual_panel(
    ax: plt.Axes,
    residual: np.ndarray,
    title: str,
    stamp_size_arcsec: float,
    vlim: float,
) -> None:
    """
    Plot a residual panel for the lensed Sersic postage stamp.
    """
    half_size = 0.5 * float(stamp_size_arcsec)
    im = ax.imshow(
        residual,
        origin="lower",
        extent=[-half_size, half_size, -half_size, half_size],
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
    )
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.25)
    ax.axvline(0.0, color="black", lw=0.5, alpha=0.25)
    ax.set_title(title, fontsize=9, pad=2)
    ax.set_xlabel(r"$\theta_1$ [arcsec]")
    ax.set_ylabel(r"$\theta_2$ [arcsec]")
    ax.set_aspect("equal")
    return im


def annotate_time_delay_diagnostic(ax: plt.Axes, stats: dict[str, float]) -> None:
    """
    Print TD refinement diagnostics inside the panel.
    """
    n_failed = int(stats.get("n_failed", 0))
    ax.text(
        0.97,
        0.03,
        f"max |Δθ| = {stats['max_shift']:.2e} arcsec\n"
        f"max |∇τ| = {stats['max_grad']:.2e}\n"
        f"med |∇τ| = {stats['median_grad']:.2e}"
        + (f"\nTD refine failed: {n_failed}" if n_failed > 0 else ""),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.25),
    )


def plot_potential_difference_figure(
    case_data: dict[str, tuple[np.ndarray, np.ndarray]],
    co: FlatLambdaCDM,
    zl: float,
    zs_norm: float,
    pixel_scale_arcsec: float,
    mask_radius_arcsec: float,
    hist_bins: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_ref = build_deflector_for_case(
        co,
        case_data["ref"][0],
        case_data["ref"][1],
        zl,
        zs_norm,
        pixel_scale_arcsec,
        compute_potential=True,
    )
    df_recon = build_deflector_for_case(
        co,
        case_data["recon"][0],
        case_data["recon"][1],
        zl,
        zs_norm,
        pixel_scale_arcsec,
        compute_potential=True,
    )

    pot_ref = np.asarray(df_ref.pot, dtype=float)
    pot_recon = np.asarray(df_recon.pot, dtype=float)

    theta_x = np.asarray(df_ref.thetax, dtype=float)
    theta_y = np.asarray(df_ref.thetay, dtype=float)
    xx, yy = np.meshgrid(theta_x, theta_y)
    rr = np.sqrt(xx**2 + yy**2)
    inside = rr <= mask_radius_arcsec
    if not np.any(inside):
        inside = np.isfinite(pot_ref) & np.isfinite(pot_recon)

    overlap = inside & np.isfinite(pot_ref) & np.isfinite(pot_recon)
    if np.any(overlap):
        potential_offset = float(np.nanmedian(pot_recon[overlap] - pot_ref[overlap]))
    else:
        potential_offset = 0.0
    pot_recon_aligned = pot_recon - potential_offset
    display_offset = float(np.nanmedian(pot_ref[inside])) if np.any(np.isfinite(pot_ref[inside])) else 0.0
    pot_ref_disp = pot_ref - display_offset
    pot_recon_disp = pot_recon_aligned - display_offset

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.divide(
            pot_recon_aligned - pot_ref,
            pot_ref,
            out=np.full_like(pot_ref, np.nan, dtype=float),
            where=np.abs(pot_ref) > np.finfo(float).eps,
        )
    rel_diff[rr > mask_radius_arcsec] = np.nan

    diff = pot_recon_aligned - pot_ref
    diff[rr > mask_radius_arcsec] = np.nan

    finite_rel = rel_diff[np.isfinite(rel_diff)]
    if finite_rel.size == 0:
        finite_rel = np.array([0.0], dtype=float)

    pot_finite = np.concatenate([
        pot_ref_disp[np.isfinite(pot_ref_disp)],
        pot_recon_disp[np.isfinite(pot_recon_disp)],
    ])
    pot_finite = pot_finite[np.isfinite(pot_finite)]
    if pot_finite.size == 0:
        pot_finite = np.array([0.0], dtype=float)

    pot_vmax = float(np.nanpercentile(np.abs(pot_finite), 99.0))
    if not np.isfinite(pot_vmax) or pot_vmax <= 0.0:
        pot_vmax = 1.0

    rel_vmax = float(np.nanpercentile(np.abs(finite_rel), 99.0))
    if not np.isfinite(rel_vmax) or rel_vmax <= 0.0:
        rel_vmax = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax_ref, ax_recon = axes[0]
    ax_diff, ax_hist = axes[1]

    im_ref = ax_ref.imshow(
        pot_ref_disp,
        origin="lower",
        extent=[theta_x[0], theta_x[-1], theta_y[0], theta_y[-1]],
        cmap="RdBu_r",
        vmin=-pot_vmax,
        vmax=pot_vmax,
    )
    ax_ref.add_patch(
        plt.Circle((0.0, 0.0), mask_radius_arcsec, fill=False, ec="white", lw=1.2, ls="--")
    )
    #ax_ref.set_title(f"REF potential (display baseline removed: {display_offset:.3g})")
    ax_ref.set_title(f"REF potential")
    ax_ref.set_xlabel(r"$\theta_1$ [arcsec]")
    ax_ref.set_ylabel(r"$\theta_2$ [arcsec]")
    ax_ref.set_aspect("equal")

    im_recon = ax_recon.imshow(
        pot_recon_disp,
        origin="lower",
        extent=[theta_x[0], theta_x[-1], theta_y[0], theta_y[-1]],
        cmap="RdBu_r",
        vmin=-pot_vmax,
        vmax=pot_vmax,
    )
    ax_recon.add_patch(
        plt.Circle((0.0, 0.0), mask_radius_arcsec, fill=False, ec="white", lw=1.2, ls="--")
    )
    ax_recon.set_title(f"RECON potential (offset corrected: {potential_offset:.3g})")
    ax_recon.set_xlabel(r"$\theta_1$ [arcsec]")
    ax_recon.set_ylabel(r"$\theta_2$ [arcsec]")
    ax_recon.set_aspect("equal")

    cbar_pot = fig.colorbar(im_ref, ax=[ax_ref, ax_recon], fraction=0.046, pad=0.02)
    cbar_pot.set_label("Potential residual")

    im_diff = ax_diff.imshow(
        rel_diff,
        origin="lower",
        extent=[theta_x[0], theta_x[-1], theta_y[0], theta_y[-1]],
        cmap="coolwarm",
        vmin=-rel_vmax,
        vmax=rel_vmax,
    )
    ax_diff.add_patch(
        plt.Circle((0.0, 0.0), mask_radius_arcsec, fill=False, ec="black", lw=1.2, ls="--")
    )
    #ax_diff.set_title(
    #    f"Relative potential difference (recon-ref)/ref\n"
    #    f"keeping r <= {mask_radius_arcsec:.1f} arcsec, zs-norm={zs_norm:.2f}"
    #)
    ax_diff.set_title(
        f"Relative potential difference (recon-ref)/ref"
    )
    ax_diff.set_xlabel(r"$\theta_1$ [arcsec]")
    ax_diff.set_ylabel(r"$\theta_2$ [arcsec]")
    ax_diff.set_aspect("equal")
    cbar_diff = fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.02)
    cbar_diff.set_label("Relative difference")

    ax_hist.hist(finite_rel, bins=hist_bins, color="tab:purple", alpha=0.85, edgecolor="white")
    ax_hist.axvline(0.0, color="black", ls="--", lw=1.2)
    ax_hist.set_title("Relative-difference distribution inside the circle")
    ax_hist.set_xlabel("(recon - ref) / ref")
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.grid(alpha=0.2)
    stats_text = (
        f"N = {finite_rel.size}\n"
        f"mean = {np.mean(finite_rel):.3g}\n"
        f"median = {np.median(finite_rel):.3g}\n"
        f"std = {np.std(finite_rel):.3g}"
    )
    ax_hist.text(
        0.97,
        0.97,
        stats_text,
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3),
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved potential-difference figure to {output_path}")
    plt.close(fig)


def plot_time_delay_examples_figure(
    case_data: dict[str, tuple[np.ndarray, np.ndarray]],
    co: FlatLambdaCDM,
    zl: float,
    zs_norm: float,
    pixel_scale_arcsec: float,
    caustics_output_path: Path,
    td_output_path: Path,
    td_images_output_path: Path,
    td_sersic_re_arcsec: float,
    td_n_rad: int,
    td_n_tan: int,
    td_zmin: float,
    td_zmax: float,
    td_seed: int,
    show_im_diagnostic: bool,
) -> None:
    """
    Plot random source examples inside caustics and the associated time-delay surfaces.
    """
    if td_n_rad < 0 or td_n_tan < 0:
        raise ValueError("td_n_rad and td_n_tan must be non-negative.")
    if td_zmax < td_zmin:
        raise ValueError("td_zmax must be >= td_zmin.")
    if td_zmin <= zl:
        td_zmin = zl + 0.05
    if td_zmax <= td_zmin:
        raise ValueError(f"Time-delay source redshift range is invalid after enforcing zl; got [{td_zmin}, {td_zmax}].")

    rng = np.random.default_rng(td_seed)
    schedule = build_time_delay_example_schedule(td_n_rad, td_n_tan, td_zmin, td_zmax)
    z_values = [z for _, z in schedule]

    fig_caustics, axes_caustics = plt.subplots(
        2,
        len(z_values),
        figsize=(4.6 * len(z_values), 7.6),
        gridspec_kw={"wspace": 0.10, "hspace": 0.14},
        squeeze=False,
    )
    fig_td, axes_td = plt.subplots(
        2,
        len(z_values),
        figsize=(4.6 * len(z_values), 7.6),
        gridspec_kw={"wspace": 0.10, "hspace": 0.14},
        squeeze=False,
    )
    fig_img, axes_img = plt.subplots(
        3,
        len(z_values),
        figsize=(4.6 * len(z_values), 7.6),
        gridspec_kw={"wspace": 0.10, "hspace": 0.14},
        squeeze=False,
    )

    stamp_size_arcsec = 120.0
    image_records: list[dict[str, object]] = []

    for col, (caustic_kind, z_target) in enumerate(schedule):
        df_ref = build_deflector_for_case(
            co,
            case_data["ref"][0],
            case_data["ref"][1],
            zl,
            zs_norm,
            pixel_scale_arcsec,
            compute_potential=True,
        )
        df_recon = build_deflector_for_case(
            co,
            case_data["recon"][0],
            case_data["recon"][1],
            zl,
            zs_norm,
            pixel_scale_arcsec,
            compute_potential=True,
        )
        df_ref.change_redshift(float(z_target))
        df_recon.change_redshift(float(z_target))
        beta, n_images_by_case, used_fallback = sample_common_source_inside_caustic_kind(
            df_ref,
            df_recon,
            caustic_kind,
            rng,
            min_images=2,
        )
        image_records.append(
            {
                "z": float(z_target),
                "kind": caustic_kind,
                "ref": build_lensed_sersic_image(
                    df_ref,
                    beta,
                    stamp_size_arcsec=stamp_size_arcsec,
                    source_re_arcsec=td_sersic_re_arcsec,
                ),
                "recon": build_lensed_sersic_image(
                    df_recon,
                    beta,
                    stamp_size_arcsec=stamp_size_arcsec,
                    source_re_arcsec=td_sersic_re_arcsec,
                ),
            }
        )

        for row_case, (case_name, df) in enumerate((("ref", df_ref), ("recon", df_recon))):
            source_ax = axes_caustics[row_case][col]
            td_ax = axes_td[row_case][col]
            short_kind = "rad" if caustic_kind == "rad" else "tan"
            plot_caustic_panel(source_ax, df, beta, f"{case_name.upper()} {short_kind}  z={z_target:.2f}")
            stats = plot_time_delay_panel(td_ax, df, beta, f"{case_name.upper()} {short_kind}  z={z_target:.2f}")
            if show_im_diagnostic:
                annotate_time_delay_diagnostic(td_ax, stats)
            print(
                f"{case_name.upper()} {short_kind} z={z_target:.2f}: "
                f"max|Δθ|={stats['max_shift']:.3e} arcsec, "
                f"max|∇τ|={stats['max_grad']:.3e}, med|∇τ|={stats['median_grad']:.3e}"
            )
            if used_fallback:
                source_ax.text(
                    0.03,
                    0.03,
                    "no caustic found\nusing fallback source",
                    transform=source_ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.25),
                )
                td_ax.text(
                    0.03,
                    0.03,
                    "no caustic found\nusing fallback source",
                    transform=td_ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.25),
                )
            #else:
            #    td_ax.text(
            #        0.03,
            #        0.03,
            #        f"accepted images: {n_images_by_case[case_name]}",
            #        transform=td_ax.transAxes,
            #       ha="left",
            #        va="bottom",
            #        fontsize=8,
            #        color="black",
            #        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.25),
            #    )

            if row_case == 0:
                source_ax.set_xlabel("")
                td_ax.set_xlabel("")
            else:
                source_ax.set_xlabel(r"$\beta_1$ [arcsec]")
                td_ax.set_xlabel(r"$\theta_1$ [arcsec]")

            if col == 0:
                source_ax.set_ylabel(r"$\beta_2$ [arcsec]")
                td_ax.set_ylabel(r"$\theta_2$ [arcsec]")
            else:
                source_ax.set_ylabel("")
                td_ax.set_ylabel("")

    # Keep the contrast fixed so changes in the Sersic effective radius remain visible
    # across repeated runs with different --td-sersic-re values.
    vmin = 1.0e-6
    vmax = 1.0
    image_norm = LogNorm(vmin=vmin, vmax=vmax)

    residual_values = []
    for rec in image_records:
        ref_img = np.asarray(rec["ref"], dtype=float)
        recon_img = np.asarray(rec["recon"], dtype=float)
        residual = recon_img - ref_img
        finite = residual[np.isfinite(residual)]
        if finite.size:
            residual_values.append(np.abs(finite))
    if residual_values:
        residual_vlim = float(np.nanpercentile(np.concatenate(residual_values), 99.0))
    else:
        residual_vlim = 1.0e-6
    if not np.isfinite(residual_vlim) or residual_vlim <= 0.0:
        residual_vlim = 1.0e-6

    for col, rec in enumerate(image_records):
        z_target = float(rec["z"])
        short_kind = "rad" if rec["kind"] == "rad" else "tan"
        for row_case, case_name in enumerate(("ref", "recon")):
            ax = axes_img[row_case][col]
            plot_lensed_sersic_panel(
                ax,
                np.asarray(rec[case_name], dtype=float),
                f"{case_name.upper()} {short_kind}  z={z_target:.2f}",
                stamp_size_arcsec=stamp_size_arcsec,
                norm=image_norm,
            )
            if row_case == 0:
                ax.set_xlabel("")
            if col > 0:
                ax.set_ylabel("")
        residual_ax = axes_img[2][col]
        residual = np.asarray(rec["recon"], dtype=float) - np.asarray(rec["ref"], dtype=float)
        plot_lensed_sersic_residual_panel(
            residual_ax,
            residual,
            f"RECON - REF  {short_kind}  z={z_target:.2f}",
            stamp_size_arcsec=stamp_size_arcsec,
            vlim=residual_vlim,
        )
        if col > 0:
            residual_ax.set_ylabel("")
        if col == 0:
            residual_ax.set_ylabel(r"$\theta_2$ [arcsec]")

    image_for_colorbar = axes_img[0][0].images[0] if axes_img.size > 0 and len(axes_img[0][0].images) > 0 else None
    fig_caustics.suptitle(
        f"Source-plane / caustic examples (N_rad={td_n_rad}, N_tan={td_n_tan}, N={td_n_rad + td_n_tan}, z in [{td_zmin:.2f}, {td_zmax:.2f}], seed={td_seed})",
        fontsize=13,
    )
    fig_td.suptitle(
        f"Time-delay surface examples (N_rad={td_n_rad}, N_tan={td_n_tan}, N={td_n_rad + td_n_tan}, z in [{td_zmin:.2f}, {td_zmax:.2f}], seed={td_seed})",
        fontsize=13,
    )
    fig_img.suptitle(
        f"Lensed Sersic source images (re={td_sersic_re_arcsec:.2f}\", q=1, size={stamp_size_arcsec:.0f}\" x {stamp_size_arcsec:.0f}\", N_rad={td_n_rad}, N_tan={td_n_tan}, N={td_n_rad + td_n_tan}, seed={td_seed})",
        fontsize=13,
    )
    caustics_output_path.parent.mkdir(parents=True, exist_ok=True)
    td_output_path.parent.mkdir(parents=True, exist_ok=True)
    td_images_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_caustics.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.885)
    fig_td.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.885)
    fig_img.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.885)
    if image_for_colorbar is not None:
        cbar_img = fig_img.colorbar(image_for_colorbar, ax=axes_img.ravel().tolist(), fraction=0.046, pad=0.02)
        cbar_img.set_label("Lensed source brightness")
    residual_for_colorbar = axes_img[2][0].images[0] if axes_img.shape[0] > 2 and len(axes_img[2][0].images) > 0 else None
    if residual_for_colorbar is not None:
        cbar_res = fig_img.colorbar(residual_for_colorbar, ax=axes_img[2].tolist(), fraction=0.046, pad=0.02)
        cbar_res.set_label("RECON - REF")
    fig_caustics.savefig(caustics_output_path, dpi=220, bbox_inches="tight")
    fig_td.savefig(td_output_path, dpi=220, bbox_inches="tight")
    fig_img.savefig(td_images_output_path, dpi=220, bbox_inches="tight")
    print(f"Saved caustic/source-plane example figure to {caustics_output_path}")
    print(f"Saved time-delay surface example figure to {td_output_path}")
    print(f"Saved lensed Sersic image example figure to {td_images_output_path}")
    plt.close(fig_caustics)
    plt.close(fig_td)
    plt.close(fig_img)


def plot_case_panel(
    ax: plt.Axes,
    df: deflector,
    zs_target: float,
    title: str,
    vmin: float,
    vmax: float,
    show_legend: bool = False,
) -> None:
    if zs_target <= df.zl:
        ax.set_facecolor("0.95")
        ax.text(
            0.5,
            0.5,
            f"z = {zs_target:.2f} is not above zl = {df.zl:.2f}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    df.change_redshift(zs_target)
    kappa = np.asarray(df.ka, dtype=float)
    theta_x = np.asarray(df.thetax, dtype=float)
    theta_y = np.asarray(df.thetay, dtype=float)
    ggsl_cs = df.ggslCrossSection(minsize=0.5)
    multima_cs = df.multImaCrossSection()

    image = ax.imshow(
        kappa,
        origin="lower",
        extent=[theta_x[0], theta_x[-1], theta_y[0], theta_y[-1]],
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
    )

    tancl = df.tancl()
    radcl = df.radcl()
    first_tan = True
    for cl in tancl:
        x, y = df.getCritPoints(cl)
        ax.plot(
            x,
            y,
            color="tab:red",
            lw=1.4,
            alpha=0.95,
            label="Tangential critical line" if show_legend and first_tan else None,
        )
        first_tan = False

    first_rad = True
    for cl in radcl:
        x, y = df.getCritPoints(cl)
        ax.plot(
            x,
            y,
            color="tab:blue",
            lw=1.1,
            ls="--",
            alpha=0.95,
            label="Radial critical line" if show_legend and first_rad else None,
        )
        first_rad = False

    cs_text = (
        f"GGSL: {ggsl_cs:.3g} arcsec^2\n"
        f"MI: {multima_cs:.3g} arcsec^2"
    )
    ax.text(
        0.03,
        0.03,
        cs_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.25),
    )

    ax.set_title(title)
    ax.set_xlabel(r"$\theta_1$ [arcsec]")
    ax.set_ylabel(r"$\theta_2$ [arcsec]")
    ax.set_aspect("equal")
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, loc="upper right", fontsize=9)
    return image


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = resolve_output_path(args.output, DEFAULT_OUTPUT.name)
    potential_output_path = resolve_output_path(args.potential_output, DEFAULT_POTENTIAL_OUTPUT.name)
    td_caustics_output_path = resolve_output_path(args.td_caustics_output, DEFAULT_TD_CAUSTICS_OUTPUT.name)
    td_output_path = resolve_output_path(args.td_output, DEFAULT_TD_OUTPUT.name)
    td_images_output_path = resolve_output_path(args.td_images_output, DEFAULT_TD_IMAGES_OUTPUT.name)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    co = FlatLambdaCDM(H0=args.h0, Om0=args.om0)
    case_data = {
        "ref": load_case(input_dir, "ref"),
        "recon": load_case(input_dir, "recon"),
    }

    zs_values = [float(z) for z in args.zs]
    if len(zs_values) == 0:
        raise ValueError("At least one source redshift is required.")
    zs_norm = float(args.zs_norm)
    if zs_norm <= args.zl:
        raise ValueError(f"zs-norm must be above zl; got zs-norm={zs_norm} and zl={args.zl}.")

    kappas_for_scaling: list[np.ndarray] = []
    for case_name, (angx, angy) in case_data.items():
        for z in zs_values:
            if z > args.zl:
                df_tmp = build_deflector_for_case(co, angx, angy, args.zl, zs_norm, args.pixel_scale_arcsec)
                df_tmp.change_redshift(z)
                kappas_for_scaling.append(np.asarray(df_tmp.ka, dtype=float))

    vmin = finite_percentile(kappas_for_scaling, 2.0) if kappas_for_scaling else 1.0
    vmax = finite_percentile(kappas_for_scaling, 98.0) if kappas_for_scaling else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(
        2,
        len(zs_values),
        figsize=(5.2 * len(zs_values), 10.0),
        constrained_layout=True,
        squeeze=False,
    )

    image_for_colorbar = None
    for row, case_name in enumerate(("ref", "recon")):
        angx, angy = case_data[case_name]
        for col, z_target in enumerate(zs_values):
            df_panel = build_deflector_for_case(
                co,
                angx,
                angy,
                args.zl,
                zs_norm,
                args.pixel_scale_arcsec,
            )
            ax = axes[row][col]
            title = f"{case_name.upper()}  z_s={z_target:.1f}"
            image = plot_case_panel(
                ax,
                df_panel,
                z_target,
                title,
                vmin,
                vmax,
                show_legend=(row == 0 and col == 0),
            )
            if image_for_colorbar is None and image is not None:
                image_for_colorbar = image

            if row == 0:
                ax.set_xlabel("")
            if col > 0:
                ax.set_ylabel("")

    if image_for_colorbar is not None:
        cbar = fig.colorbar(image_for_colorbar, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
        cbar.set_label(r"$\kappa$")

    fig.suptitle(
        f"Deflection Maps: reference vs reconstructed  (zl={args.zl:.3f}, zs-norm={zs_norm:.2f}, pixel scale={args.pixel_scale_arcsec:.4f} arcsec/pix)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {output_path}")

    plot_potential_difference_figure(
        case_data=case_data,
        co=co,
        zl=args.zl,
        zs_norm=zs_norm,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        mask_radius_arcsec=float(args.potential_mask_radius),
        hist_bins=int(args.potential_hist_bins),
        output_path=potential_output_path,
    )

    if args.show_td_examples:
        plot_time_delay_examples_figure(
            case_data=case_data,
            co=co,
            zl=args.zl,
            zs_norm=zs_norm,
            pixel_scale_arcsec=args.pixel_scale_arcsec,
            caustics_output_path=td_caustics_output_path,
            td_output_path=td_output_path,
            td_images_output_path=td_images_output_path,
            td_sersic_re_arcsec=float(args.td_sersic_re),
            td_n_rad=int(args.td_n_rad),
            td_n_tan=int(args.td_n_tan),
            td_zmin=float(args.td_zmin),
            td_zmax=float(args.td_zmax),
            td_seed=int(args.td_seed),
            show_im_diagnostic=bool(args.show_im_diagnostic),
        )

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
