#!/usr/bin/env python3
"""
Measure time-delay statistics for many randomly sampled multiply-imaged sources.

The script samples source redshifts in a user-specified range, places the
sources inside the REF caustics, computes matched image positions for the REF
and RECON models, and compares the relative time delays of the matched images.
The output is a histogram of the RECON-minus-REF time-delay differences.

Example
-------
python apps/xincheng/td_statistics.py \
    --zl 0.5 \
    --pixel-scale-arcsec 0.6361 \
    --n-rad 1 \
    --n-tan 4 \
    --zsmin 1.0 \
    --zsmax 3.0
"""

from __future__ import annotations

import argparse
import sys
import string
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyLensLib.deflector import deflector
from pyLensLib.pointsrc import pointsrc


DEFAULT_INPUT_DIR = Path("/Users/maxmen3/Downloads/deflection_outputs")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "td_statistics_histogram.png"


def resolve_output_path(output_path: Path, default_filename: str) -> Path:
    output_path = output_path.expanduser().resolve()
    if (output_path.exists() and output_path.is_dir()) or output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / default_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute time-delay statistics for many multiply imaged sources."
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
        help=f"Histogram output path or directory. Default: {DEFAULT_OUTPUT}",
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
        "--n-rad",
        type=int,
        default=1,
        help="Number of sources to draw inside the radial caustic. Default: 1",
    )
    parser.add_argument(
        "--n-tan",
        type=int,
        default=4,
        help="Number of sources to draw inside the tangential caustic. Default: 4",
    )
    parser.add_argument(
        "--zsmin",
        type=float,
        default=1.0,
        help="Lower bound of the random source-redshift range. Default: 1.0",
    )
    parser.add_argument(
        "--zsmax",
        type=float,
        default=3.0,
        help="Upper bound of the random source-redshift range. Default: 3.0",
    )
    parser.add_argument(
        "--match-max-sep-arcsec",
        type=float,
        default=2.0,
        help="Maximum separation allowed when matching REF and RECON images. Default: 2.0",
    )
    parser.add_argument(
        "--discard-mismatched-image-counts",
        action="store_true",
        default=False,
        help=(
            "Deprecated: sources with mismatched REF/RECON image counts are always "
            "skipped."
        ),
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=60,
        help="Number of histogram bins. Default: 60",
    )
    parser.add_argument(
        "--clipped-source-example-count",
        type=int,
        default=4,
        help="Number of clipped-source examples to plot. Default: 4",
    )
    parser.add_argument(
        "--min-ref-delay-days",
        type=float,
        default=1.0,
        help="Exclude matched images with REF delay below this threshold in days. Default: 1.0",
    )
    parser.add_argument(
        "--discard-clipped-sources",
        action="store_true",
        default=False,
        help=(
            "Exclude relative TD values outside the 1st-99th percentile plotting range "
            "from the summary statistics. Default: False"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for source placement and redshift sampling. Default: 12345",
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show the histogram interactively after saving.",
    )
    show_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Only save the histogram.",
    )
    parser.set_defaults(show=False)
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
    compute_potential: bool = True,
) -> deflector:
    df = deflector(co, angx=angx, angy=angy, zl=zl, zs=zs_norm)
    theta = build_theta_grid(angx.shape[0], pixel_scale_arcsec)
    df.setGrid(theta=theta, compute_potential=compute_potential)
    return df


def sample_point_in_geometry(
    geometry,
    rng: np.random.Generator,
    max_tries: int = 10000,
    fallback: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
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


def caustic_union_by_kind(df: deflector, caustic_kind: str):
    kind = caustic_kind.lower()
    if kind not in {"rad", "tan"}:
        raise ValueError(f"Unknown caustic kind: {caustic_kind}")
    cl = df.radcl() if kind == "rad" else df.tancl()
    caustics = df.getCaustics(cl)
    geoms = [c.geometria for c in caustics if not c.geometria.is_empty]
    if not geoms:
        return unary_union([])
    return unary_union(geoms)


def sample_source_inside_ref_caustic_kind(
    df_ref: deflector,
    df_recon: deflector,
    caustic_kind: str,
    rng: np.random.Generator,
    min_images: int = 2,
    max_tries: int = 1000,
) -> tuple[tuple[float, float], bool, int, int]:
    """
    Sample a source inside the REF caustics and require multiply-imaged solutions
    in both REF and RECON.
    """
    ref_union = caustic_union_by_kind(df_ref, caustic_kind)
    pixel_fallback = ((df_ref.nray1 - 1) / 2.0, (df_ref.nray2 - 1) / 2.0)

    def evaluate(beta_pix: tuple[float, float]) -> tuple[tuple[float, float], int, int]:
        beta = (
            beta_pix[0] * df_ref.pixel_scale + df_ref.thetax[0],
            beta_pix[1] * df_ref.pixel_scale + df_ref.thetay[0],
        )
        ps_ref = build_point_source(df_ref, beta)
        ps_recon = build_point_source(df_recon, beta)
        return beta, int(len(ps_ref.xi1)), int(len(ps_recon.xi1))

    for _ in range(max_tries):
        beta_pix = sample_point_in_geometry(ref_union, rng, fallback=pixel_fallback)
        beta, n_ref, n_recon = evaluate(beta_pix)
        if n_ref >= min_images and n_recon >= min_images:
            return beta, False, n_ref, n_recon

    if not ref_union.is_empty:
        beta_pix = ref_union.representative_point()
    else:
        beta_pix = Point(pixel_fallback[0], pixel_fallback[1])
    beta, n_ref, n_recon = evaluate((float(beta_pix.x), float(beta_pix.y)))
    return beta, True, n_ref, n_recon


def build_point_source(
    df: deflector,
    beta: tuple[float, float],
    td_rel: np.ndarray | None = None,
    grad_x: np.ndarray | None = None,
    grad_y: np.ndarray | None = None,
) -> pointsrc:
    if td_rel is None or grad_x is None or grad_y is None:
        td = df.t_delay_surf(beta=beta)
        td_rel = td - np.nanmin(td)
        grad_y, grad_x = np.gradient(td_rel, df.pixel_scale, df.pixel_scale)
    return pointsrc(
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


def compute_image_eigen_signs(
    df: deflector,
    xi: np.ndarray,
    yi: np.ndarray,
) -> np.ndarray:
    """
    Return the sign of the two Jacobian eigenvalues at the image positions.

    The eigenvalues are taken as the usual tangential/radial pair
    1 - kappa ± gamma, evaluated at the image locations.
    """
    xi = np.asarray(xi, dtype=float)
    yi = np.asarray(yi, dtype=float)
    if xi.size == 0:
        return np.zeros((0, 2), dtype=int)

    xpix = xi / df.pixel_scale + (df.nray1 - 1) / 2.0
    ypix = yi / df.pixel_scale + (df.nray2 - 1) / 2.0
    kappa = np.asarray(map_coordinates(df.ka, [ypix, xpix], order=1, mode="nearest"), dtype=float)
    g1 = np.asarray(map_coordinates(df.g1, [ypix, xpix], order=1, mode="nearest"), dtype=float)
    g2 = np.asarray(map_coordinates(df.g2, [ypix, xpix], order=1, mode="nearest"), dtype=float)
    gamma = np.hypot(g1, g2)
    lam_t = 1.0 - kappa - gamma
    lam_r = 1.0 - kappa + gamma

    def signed(arr: np.ndarray) -> np.ndarray:
        out = np.sign(arr).astype(int)
        out[np.abs(arr) <= 1e-10] = 0
        return out

    return np.stack([signed(lam_t), signed(lam_r)], axis=1)


def compute_relative_td(
    df: deflector,
    beta: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return image positions and relative delays for a single model.

    The delays are shifted so that the earliest-arriving image has zero delay.
    """
    td = df.t_delay_surf(beta=beta)
    td_rel = td - np.nanmin(td)
    grad_y, grad_x = np.gradient(td_rel, df.pixel_scale, df.pixel_scale)
    ps = build_point_source(df, beta, td_rel=td_rel, grad_x=grad_x, grad_y=grad_y)
    xi = np.asarray(ps.xi1, dtype=float)
    yi = np.asarray(ps.xi2, dtype=float)
    if xi.size == 0:
        return xi, yi, np.array([], dtype=float), np.zeros((0, 2), dtype=int)
    xpix = xi / df.pixel_scale + (df.nray1 - 1) / 2.0
    ypix = yi / df.pixel_scale + (df.nray2 - 1) / 2.0
    td_img = np.asarray(map_coordinates(td_rel, [ypix, xpix], order=1, mode="nearest"), dtype=float)
    finite = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(td_img)
    xi = xi[finite]
    yi = yi[finite]
    td_img = td_img[finite]
    parity = compute_image_eigen_signs(df, xi, yi)
    if td_img.size == 0:
        return xi, yi, td_img, parity
    td_img = td_img - float(np.nanmin(td_img))
    return xi, yi, td_img, parity


def match_images_by_position(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_td: np.ndarray,
    recon_x: np.ndarray,
    recon_y: np.ndarray,
    recon_td: np.ndarray,
    max_sep_arcsec: float,
    ref_parity: np.ndarray | None = None,
    recon_parity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match REF and RECON images by minimizing image-plane separation.

    If parity signatures are supplied, only images with matching Jacobian
    eigenvalue signs are allowed to pair.
    """
    ref_x = np.asarray(ref_x, dtype=float)
    ref_y = np.asarray(ref_y, dtype=float)
    ref_td = np.asarray(ref_td, dtype=float)
    recon_x = np.asarray(recon_x, dtype=float)
    recon_y = np.asarray(recon_y, dtype=float)
    recon_td = np.asarray(recon_td, dtype=float)
    if ref_parity is not None:
        ref_parity = np.asarray(ref_parity, dtype=int)
    if recon_parity is not None:
        recon_parity = np.asarray(recon_parity, dtype=int)

    if ref_x.size == 0 or recon_x.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=float), np.array([], dtype=float)

    dist = np.hypot(ref_x[:, None] - recon_x[None, :], ref_y[:, None] - recon_y[None, :])
    if ref_parity is not None and recon_parity is not None:
        if ref_parity.shape[0] != ref_x.size or recon_parity.shape[0] != recon_x.size:
            raise ValueError("Parity arrays must match the number of images in each model.")
        parity_match = np.all(ref_parity[:, None, :] == recon_parity[None, :, :], axis=-1)
        if np.any(np.isfinite(dist)):
            finite_cost = dist[np.isfinite(dist)]
            max_cost = float(np.nanmax(finite_cost)) if finite_cost.size else 1.0
        else:
            max_cost = 1.0
        penalty = max(max_cost, 1.0) * 1e6 + max_cost
        dist = np.where(parity_match, dist, penalty)
    row_ind, col_ind = linear_sum_assignment(dist)
    matched = dist[row_ind, col_ind] <= float(max_sep_arcsec)
    if not np.any(matched):
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=float), np.array([], dtype=float)
    row_sel = row_ind[matched]
    col_sel = col_ind[matched]
    abs_diffs = recon_td[col_sel] - ref_td[row_sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diffs = np.divide(
            abs_diffs,
            ref_td[row_sel],
            out=np.full_like(abs_diffs, np.nan, dtype=float),
            where=np.abs(ref_td[row_sel]) > np.finfo(float).eps,
        )
    return row_sel, col_sel, np.asarray(abs_diffs, dtype=float), np.asarray(rel_diffs, dtype=float)


def match_images_by_td_order(
    ref_td: np.ndarray,
    recon_td: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match the already position-matched images by sorting each case in
    increasing arrival-time order and pairing equal ranks.
    """
    ref_td = np.asarray(ref_td, dtype=float)
    recon_td = np.asarray(recon_td, dtype=float)
    if ref_td.size == 0 or recon_td.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    ref_sorted = np.sort(ref_td)
    recon_sorted = np.sort(recon_td)
    n_match = min(ref_sorted.size, recon_sorted.size)
    if n_match == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    ref_sorted = ref_sorted[:n_match]
    recon_sorted = recon_sorted[:n_match]
    abs_diffs = recon_sorted - ref_sorted
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diffs = np.divide(
            abs_diffs,
            ref_sorted,
            out=np.full_like(abs_diffs, np.nan, dtype=float),
            where=np.abs(ref_sorted) > np.finfo(float).eps,
        )
    return np.asarray(abs_diffs, dtype=float), np.asarray(rel_diffs, dtype=float)


def sort_matched_by_arrival_time(
    ref_td: np.ndarray,
    recon_td: np.ndarray,
    ref_indices: np.ndarray,
    recon_indices: np.ndarray,
    ref_parity: np.ndarray | None = None,
    recon_parity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reorder a position-matched image set by increasing REF arrival time.

    If parity arrays are supplied, the reordered matched pairs are validated to
    preserve the same Jacobian eigenvalue-sign signature on both sides.
    """
    ref_td = np.asarray(ref_td, dtype=float)
    recon_td = np.asarray(recon_td, dtype=float)
    ref_indices = np.asarray(ref_indices, dtype=int)
    recon_indices = np.asarray(recon_indices, dtype=int)
    if ref_parity is not None:
        ref_parity = np.asarray(ref_parity, dtype=int)
    if recon_parity is not None:
        recon_parity = np.asarray(recon_parity, dtype=int)
    if ref_td.size == 0 or recon_td.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=float), np.array([], dtype=float)
    if ref_parity is not None and recon_parity is not None:
        if ref_parity.shape != recon_parity.shape or ref_parity.shape[0] != ref_td.size:
            raise ValueError("Parity arrays must match the size of the matched image set.")

    order = np.argsort(ref_td)
    ref_td = ref_td[order]
    recon_td = recon_td[order]
    ref_indices = ref_indices[order]
    recon_indices = recon_indices[order]
    if ref_parity is not None and recon_parity is not None:
        ref_parity = ref_parity[order]
        recon_parity = recon_parity[order]
        if not np.array_equal(ref_parity, recon_parity):
            raise RuntimeError(
                "Arrival-time ordering is not parity-consistent with the position-matched set."
            )
    abs_diffs = recon_td - ref_td
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diffs = np.divide(
            abs_diffs,
            ref_td,
            out=np.full_like(abs_diffs, np.nan, dtype=float),
            where=np.abs(ref_td) > np.finfo(float).eps,
        )
    return ref_indices, recon_indices, np.asarray(abs_diffs, dtype=float), np.asarray(rel_diffs, dtype=float)


def prepare_relative_sample(
    rel_diffs: np.ndarray,
    discard_clipped_sources: bool,
) -> tuple[np.ndarray, float, float, float, float, float, float, str, int]:
    rel_valid, rel_lo, rel_hi = compute_relative_clip_bounds(rel_diffs)
    rel_clipped_mask = (rel_valid >= rel_lo) & (rel_valid <= rel_hi)
    rel_clipped = int(np.count_nonzero(~rel_clipped_mask))
    rel_plot = rel_valid[rel_clipped_mask]
    if rel_plot.size == 0:
        rel_plot = rel_valid
        rel_clipped = 0
    analysis_rel = rel_plot if discard_clipped_sources else rel_valid
    analysis_label = "clipped" if discard_clipped_sources else "full"
    rel_mean = float(np.mean(analysis_rel))
    rel_median = float(np.median(analysis_rel))
    rel_rms = float(np.sqrt(np.mean(analysis_rel**2)))
    return rel_plot, rel_lo, rel_hi, rel_mean, rel_median, rel_rms, analysis_label, rel_clipped


def compute_relative_clip_bounds(rel_diffs: np.ndarray) -> tuple[np.ndarray, float, float]:
    rel_valid = np.asarray(rel_diffs, dtype=float)
    rel_valid = rel_valid[np.isfinite(rel_valid)]
    if rel_valid.size == 0:
        rel_valid = np.array([0.0], dtype=float)
    rel_lo, rel_hi = np.nanpercentile(rel_valid, [1.0, 99.0])
    if not np.isfinite(rel_lo) or not np.isfinite(rel_hi) or rel_lo == rel_hi:
        rel_lo = float(np.nanmin(rel_valid)) if np.any(np.isfinite(rel_valid)) else -1.0
        rel_hi = float(np.nanmax(rel_valid)) if np.any(np.isfinite(rel_valid)) else 1.0
        if rel_lo == rel_hi:
            rel_lo -= 1.0
            rel_hi += 1.0
    return rel_valid, float(rel_lo), float(rel_hi)


def plot_clipped_source_examples(
    records: list[dict[str, object]],
    rel_lo: float,
    rel_hi: float,
    output_path: Path,
    max_examples: int,
    min_ref_delay_days: float,
) -> None:
    clipped_records: list[dict[str, object]] = []
    for rec in records:
        ref_td = np.asarray(rec["ref_td"], dtype=float)
        rel_diffs = np.asarray(rec["rel_diffs"], dtype=float)
        row_sel = np.asarray(rec["row_sel"], dtype=int)
        used = np.isfinite(rel_diffs) & np.isfinite(ref_td[row_sel]) & (ref_td[row_sel] >= float(min_ref_delay_days))
        clipped = used & ((rel_diffs < rel_lo) | (rel_diffs > rel_hi))
        clipped_count = int(np.count_nonzero(clipped))
        if clipped_count > 0:
            rec = dict(rec)
            rec["clipped_count"] = clipped_count
            rec["clipped_mask"] = clipped
            clipped_records.append(rec)

    if not clipped_records:
        print("No clipped-source examples to plot.")
        return

    clipped_records.sort(key=lambda rec: (int(rec["clipped_count"]), float(rec["z"])), reverse=True)
    examples = clipped_records[: max(1, int(max_examples))]
    n_panels = len(examples)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 6.0 * nrows), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    letters = string.ascii_uppercase
    for idx, rec in enumerate(examples):
        ax = axes.flat[idx]
        ax.set_visible(True)
        ref_x = np.asarray(rec["ref_x"], dtype=float)
        ref_y = np.asarray(rec["ref_y"], dtype=float)
        recon_x = np.asarray(rec["recon_x"], dtype=float)
        recon_y = np.asarray(rec["recon_y"], dtype=float)
        ref_td = np.asarray(rec["ref_td"], dtype=float)
        row_sel = np.asarray(rec["row_sel"], dtype=int)
        col_sel = np.asarray(rec["col_sel"], dtype=int)
        rel_diffs = np.asarray(rec["rel_diffs"], dtype=float)
        clipped_mask = np.asarray(rec["clipped_mask"], dtype=bool)
        order = np.argsort(ref_td[row_sel])
        row_sel = row_sel[order]
        col_sel = col_sel[order]
        clipped_mask = clipped_mask[order]
        ref_used_x = ref_x[row_sel]
        ref_used_y = ref_y[row_sel]
        recon_used_x = recon_x[col_sel]
        recon_used_y = recon_y[col_sel]
        for j in range(row_sel.size):
            line_color = "tab:red" if clipped_mask[j] else "0.75"
            line_style = "--" if clipped_mask[j] else "-"
            ax.plot(
                [ref_used_x[j], recon_used_x[j]],
                [ref_used_y[j], recon_used_y[j]],
                color=line_color,
                lw=1.0,
                ls=line_style,
                alpha=0.8 if clipped_mask[j] else 0.5,
                zorder=1,
            )
        ax.scatter(
            ref_used_x,
            ref_used_y,
            s=60,
            marker="o",
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=1.5,
            label="REF images",
            zorder=3,
        )
        ax.scatter(
            recon_used_x,
            recon_used_y,
            s=60,
            marker="s",
            facecolors="none",
            edgecolors="tab:orange",
            linewidths=1.5,
            label="RECON images",
            zorder=3,
        )
        if row_sel.size > 0:
            ax.scatter(
                ref_used_x[0],
                ref_used_y[0],
                s=130,
                marker="*",
                color="black",
                edgecolors="black",
                zorder=4,
                label="A image",
            )
            ax.scatter(
                recon_used_x[0],
                recon_used_y[0],
                s=130,
                marker="*",
                color="black",
                edgecolors="black",
                zorder=4,
            )
        if np.any(clipped_mask):
            ax.scatter(
                ref_used_x[clipped_mask],
                ref_used_y[clipped_mask],
                s=180,
                marker="*",
                facecolors="none",
                edgecolors="tab:red",
                linewidths=1.8,
                zorder=5,
                label="clipped pair",
            )
            ax.scatter(
                recon_used_x[clipped_mask],
                recon_used_y[clipped_mask],
                s=180,
                marker="*",
                facecolors="none",
                edgecolors="tab:red",
                linewidths=1.8,
                zorder=5,
            )
        for j in range(row_sel.size):
            letter = letters[j % len(letters)]
            ax.text(ref_used_x[j], ref_used_y[j], letter, fontsize=9, ha="right", va="bottom", color="tab:blue")
            ax.text(recon_used_x[j], recon_used_y[j], letter, fontsize=9, ha="left", va="bottom", color="tab:orange")
        ax.axhline(0.0, color="0.85", lw=0.8)
        ax.axvline(0.0, color="0.85", lw=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\theta_1$ [arcsec]")
        ax.set_ylabel(r"$\theta_2$ [arcsec]")
        ax.set_title(
            f"{rec['kind']} z={float(rec['z']):.2f}  clipped={int(rec['clipped_count'])}",
            fontsize=11,
        )
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="tab:blue", markersize=8, label="REF images"),
        Line2D([], [], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor="tab:orange", markersize=8, label="RECON images"),
        Line2D([], [], marker="*", linestyle="None", color="black", markersize=10, label="A image"),
        Line2D([], [], marker="*", linestyle="None", color="tab:red", markersize=10, label="clipped pair"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False)

    fig.suptitle("Clipped source-image examples: REF vs RECON", y=0.98, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved clipped-source diagnostics to {output_path}")


def plot_histograms(
    abs_diffs: np.ndarray,
    rel_diffs: np.ndarray,
    output_path: Path,
    hist_bins: int,
    show: bool,
    discard_clipped_sources: bool,
    n_sources: int,
    pos_rms_arcsec: float,
    n_pos_pairs: int,
    n_pairs_used: int,
    n_ref_delay_excluded: int,
    min_ref_delay_days: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    abs_mean = float(np.mean(abs_diffs))
    abs_median = float(np.median(abs_diffs))
    abs_rms = float(np.sqrt(np.mean(abs_diffs**2)))

    rel_plot, rel_lo, rel_hi, rel_mean, rel_median, rel_rms, analysis_label, rel_clipped = prepare_relative_sample(
        rel_diffs, discard_clipped_sources
    )
    rel_bins = np.linspace(rel_lo, rel_hi, int(hist_bins) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    ax_abs_pos, ax_rel_pos = axes

    def render_abs(ax: plt.Axes, data: np.ndarray, mean: float, median: float, rms: float, title: str, xlabel: str, n_pairs: int) -> None:
        ax.hist(data, bins=hist_bins, color="tab:blue", alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="black", ls="--", lw=1.2)
        ax.axvline(mean, color="tab:red", lw=1.4, label=f"mean = {mean:.3f} d")
        ax.axvline(median, color="tab:green", lw=1.4, label=f"median = {median:.3f} d")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of matched images")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="best")
        ax.text(
            0.03,
            0.97,
            f"sources = {n_sources}\n"
            f"positional RMS = {pos_rms_arcsec:.4f} arcsec\n"
            f"positional pairs = {n_pos_pairs}\n"
            f"matched images used = {n_pairs}\n"
            f"excluded if $\\Delta t_\\mathrm{{REF}}$ < {min_ref_delay_days:.1f} d = {n_ref_delay_excluded}\n"
            f"mean = {mean:.3f} d\n"
            f"median = {median:.3f} d\n"
            f"rms = {rms:.3f} d",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
        )

    def render_rel(
        ax: plt.Axes,
        data_plot: np.ndarray,
        bins: np.ndarray,
        mean: float,
        median: float,
        rms: float,
        title: str,
        xlabel: str,
        n_pairs: int,
        rel_clipped_n: int,
        analysis_label_here: str,
        xlim: tuple[float, float],
    ) -> None:
        ax.hist(data_plot, bins=bins, color="tab:orange", alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="black", ls="--", lw=1.2)
        ax.axvline(mean, color="tab:red", lw=1.4, label=f"mean = {mean:.3f}")
        ax.axvline(median, color="tab:green", lw=1.4, label=f"median = {median:.3f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of matched images")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="best")
        ax.set_xlim(*xlim)
        ax.text(
            0.03,
            0.97,
            f"sources = {n_sources}\n"
            f"positional RMS = {pos_rms_arcsec:.4f} arcsec\n"
            f"positional pairs = {n_pos_pairs}\n"
            f"matched images used = {n_pairs}\n"
            f"excluded if $\\Delta t_\\mathrm{{REF}}$ < {min_ref_delay_days:.1f} d = {n_ref_delay_excluded}\n"
            f"analysis = {analysis_label_here}\n"
            f"mean = {mean:.3f}\n"
            f"median = {median:.3f}\n"
            f"rms = {rms:.3f}\n"
            f"clipped = {rel_clipped_n}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
        )

    render_abs(
        ax_abs_pos,
        abs_diffs,
        abs_mean,
        abs_median,
        abs_rms,
        "Relative TD differences by image position+parity: RECON - REF",
        r"$\Delta t_{\rm RECON,rel} - \Delta t_{\rm REF,rel}$ [days]",
        n_pairs_used,
    )
    render_rel(
        ax_rel_pos,
        rel_plot,
        rel_bins,
        rel_mean,
        rel_median,
        rel_rms,
        "Fractional relative TD differences by image position+parity",
        r"$\left(\Delta t_{\rm RECON,rel} - \Delta t_{\rm REF,rel}\right) / \Delta t_{\rm REF,rel}$",
        n_pairs_used,
        rel_clipped,
        analysis_label,
        (rel_lo, rel_hi),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved histogram to {output_path}")
    print(f"Position+parity positional RMS (all matched images, including A): {pos_rms_arcsec:.6f} arcsec")
    print(f"Position+parity positional matched-pair count: {n_pos_pairs}")
    print(f"Position+parity relative TD difference mean: {abs_mean:.6f} d")
    print(f"Position+parity relative TD difference median: {abs_median:.6f} d")
    print(
        "Position+parity relative TD RMS "
        f"(Δt_RECON,rel - Δt_REF,rel): {abs_rms:.6f} d"
    )
    print(f"Position+parity fractional relative TD analysis sample: {analysis_label}")
    print(f"Position+parity fractional relative TD difference mean: {rel_mean:.6f}")
    print(f"Position+parity fractional relative TD difference median: {rel_median:.6f}")
    print(f"Position+parity fractional relative TD RMS: {rel_rms:.6f}")
    print(f"Position+parity relative TD differences clipped from plot: {rel_clipped}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_statistics(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.expanduser().resolve()
    output_path = resolve_output_path(args.output, DEFAULT_OUTPUT.name)
    if args.zsmax < args.zsmin:
        raise ValueError("--zsmax must be >= --zsmin")
    if args.zsmin <= args.zl:
        args.zsmin = args.zl + 0.05
    if args.zsmax <= args.zsmin:
        raise ValueError("Source redshift range is invalid after enforcing zs > zl.")

    rng = np.random.default_rng(args.seed)
    co = FlatLambdaCDM(H0=float(args.h0), Om0=float(args.om0))
    ref_ax, ref_ay = load_case(input_dir, "ref")
    recon_ax, recon_ay = load_case(input_dir, "recon")
    kinds = ["rad"] * int(args.n_rad) + ["tan"] * int(args.n_tan)
    if not kinds:
        raise ValueError("At least one of --n-rad or --n-tan must be positive.")
    z_samples = rng.uniform(float(args.zsmin), float(args.zsmax), size=len(kinds))

    abs_diffs_all: list[np.ndarray] = []
    rel_diffs_all: list[np.ndarray] = []
    total_pairs_used = 0
    total_ref_delay_excluded = 0
    total_pos_pairs = 0
    total_pos_sq = 0.0
    total_sources = 0
    skipped_count = 0
    fallback_count = 0
    source_records: list[dict[str, object]] = []

    for idx, (kind, z_target) in enumerate(zip(kinds, z_samples), start=1):
        df_ref = build_deflector_for_case(
            co,
            ref_ax,
            ref_ay,
            float(args.zl),
            float(args.zs_norm),
            float(args.pixel_scale_arcsec),
            compute_potential=True,
        )
        df_recon = build_deflector_for_case(
            co,
            recon_ax,
            recon_ay,
            float(args.zl),
            float(args.zs_norm),
            float(args.pixel_scale_arcsec),
            compute_potential=True,
        )
        df_ref.change_redshift(float(z_target))
        df_recon.change_redshift(float(z_target))

        beta, used_fallback, n_ref, n_recon = sample_source_inside_ref_caustic_kind(
            df_ref,
            df_recon,
            kind,
            rng,
            min_images=2,
        )
        fallback_count += int(used_fallback)
        total_sources += 1

        if n_ref != n_recon:
            skipped_count += 1
            print(
                f"[{idx}/{len(kinds)}] kind={kind} z={z_target:.3f}: "
                f"skipped because ref_images={n_ref} != recon_images={n_recon}"
            )
            continue

        ref_x, ref_y, ref_td, ref_parity = compute_relative_td(df_ref, beta)
        recon_x, recon_y, recon_td, recon_parity = compute_relative_td(df_recon, beta)
        row_sel, col_sel, abs_diffs, rel_diffs = match_images_by_position(
            ref_x,
            ref_y,
            ref_td,
            recon_x,
            recon_y,
            recon_td,
            max_sep_arcsec=float(args.match_max_sep_arcsec),
            ref_parity=ref_parity,
            recon_parity=recon_parity,
        )
        if abs_diffs.size == 0:
            print(
                f"[{idx}/{len(kinds)}] kind={kind} z={z_target:.3f}: "
                f"no matched image pairs within {float(args.match_max_sep_arcsec):.2f} arcsec "
                f"after parity filtering"
            )
            continue
        pos_sep = np.hypot(ref_x[row_sel] - recon_x[col_sel], ref_y[row_sel] - recon_y[col_sel])
        if pos_sep.size > 0:
            total_pos_pairs += int(pos_sep.size)
            total_pos_sq += float(np.sum(pos_sep**2))
        ref_delay = np.asarray(ref_td[row_sel], dtype=float)
        used = np.isfinite(abs_diffs) & np.isfinite(rel_diffs)
        used &= np.isfinite(ref_delay)
        used &= ref_delay >= float(args.min_ref_delay_days)
        used_count = int(np.count_nonzero(used))
        if used_count == 0:
            print(
                f"[{idx}/{len(kinds)}] kind={kind} z={z_target:.3f}: "
                f"matched pairs found but none with REF delay >= {float(args.min_ref_delay_days):.2f} d"
            )
            continue
        total_ref_delay_excluded += int(abs_diffs.size - used_count)
        abs_diffs_all.append(np.asarray(abs_diffs[used], dtype=float))
        rel_diffs_all.append(np.asarray(rel_diffs[used], dtype=float))
        total_pairs_used += used_count
        source_records.append(
            {
                "kind": kind,
                "z": float(z_target),
                "beta": beta,
                "ref_x": np.asarray(ref_x, dtype=float),
                "ref_y": np.asarray(ref_y, dtype=float),
                "recon_x": np.asarray(recon_x, dtype=float),
                "recon_y": np.asarray(recon_y, dtype=float),
                "ref_td": np.asarray(ref_td, dtype=float),
                "recon_td": np.asarray(recon_td, dtype=float),
                "row_sel": np.asarray(row_sel, dtype=int),
                "col_sel": np.asarray(col_sel, dtype=int),
                "abs_diffs": np.asarray(abs_diffs, dtype=float),
                "rel_diffs": np.asarray(rel_diffs, dtype=float),
            }
        )
        print(
            f"[{idx}/{len(kinds)}] kind={kind} z={z_target:.3f}: "
            f"matched_pos={abs_diffs.size}, used={used_count}, "
            f"ref_images={len(ref_x)}, recon_images={len(recon_x)}"
        )

    if not abs_diffs_all:
        raise RuntimeError("No matched time-delay differences were collected.")

    all_abs_diffs = np.concatenate(abs_diffs_all)
    all_rel_diffs = np.concatenate(rel_diffs_all) if rel_diffs_all else np.array([], dtype=float)
    pos_rms_arcsec = float(np.sqrt(total_pos_sq / total_pos_pairs)) if total_pos_pairs > 0 else float("nan")
    _, rel_lo_global, rel_hi_global = compute_relative_clip_bounds(all_rel_diffs)
    plot_histograms(
        abs_diffs=all_abs_diffs,
        rel_diffs=all_rel_diffs,
        output_path=output_path,
        hist_bins=int(args.hist_bins),
        show=bool(args.show),
        discard_clipped_sources=bool(args.discard_clipped_sources),
        n_sources=total_sources,
        pos_rms_arcsec=pos_rms_arcsec,
        n_pos_pairs=total_pos_pairs,
        n_pairs_used=total_pairs_used,
        n_ref_delay_excluded=total_ref_delay_excluded,
        min_ref_delay_days=float(args.min_ref_delay_days),
    )
    clipped_diag_path = output_path.with_name(f"{output_path.stem}_clipped_sources.png")
    plot_clipped_source_examples(
        source_records,
        rel_lo=rel_lo_global,
        rel_hi=rel_hi_global,
        output_path=clipped_diag_path,
        max_examples=int(args.clipped_source_example_count),
        min_ref_delay_days=float(args.min_ref_delay_days),
    )
    print(f"Fallback sources used: {fallback_count}")
    print(f"Sources skipped for mismatched image counts: {skipped_count}")


def main() -> None:
    args = parse_args()
    run_statistics(args)


if __name__ == "__main__":
    main()
