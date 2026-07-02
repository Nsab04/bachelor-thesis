#!/usr/bin/env python3
"""
Sample sources from an empirical source-redshift distribution, ray-trace them
through the REF and RECON lens models, and visualize the matched-image
displacements.

The script:
  - bootsraps source redshifts from an empirical sample distribution
  - samples source positions inside the REF caustics at each sampled redshift
  - computes multiple images for REF and RECON with image-position refinement
  - matches images by position and parity of the Jacobian eigenvalues
  - plots:
      left: REF convergence map at zs=1.6 with critical lines and all images
      right: matched-image displacement sticks in the theta1/theta2 plane

By default the source-redshift samples are taken from the LTsimcat audit table
stored at:

  /Users/maxmen3/projects/pyLensLib/apps/LTsimcat/ltsimcat_output/source_caustic_audit.csv

using the `zgal` column. This is an empirical distribution that can be replaced
with a different sample file via the command line.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "sampled_image_displacements.png"
DEFAULT_ZS_SAMPLES = Path("/Users/maxmen3/projects/pyLensLib/apps/LTsimcat/ltsimcat_output/source_caustic_audit.csv")
DEFAULT_ZS_COLUMN = "zgal"
DEFAULT_DISPLAY_ZS = 1.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample sources from an empirical redshift distribution and plot image displacements."
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
        "--sources-csv",
        type=Path,
        default=None,
        help="Optional CSV output path or directory for the accepted source catalog.",
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
        "--display-zs",
        type=float,
        default=DEFAULT_DISPLAY_ZS,
        help="Source redshift used to display the REF critical lines. Default: 1.6",
    )
    parser.add_argument(
        "--h0",
        type=float,
        default=70.0,
        help="Hubble constant for the cosmology. Default: 70.0",
    )
    parser.add_argument(
        "--om0",
        type=float,
        default=0.3,
        help="Matter density for the cosmology. Default: 0.3",
    )
    parser.add_argument(
        "--n-sources",
        type=int,
        default=30,
        help="Number of sources to generate. Default: 30",
    )
    parser.add_argument(
        "--zs-samples-file",
        type=Path,
        default=DEFAULT_ZS_SAMPLES,
        help=(
            "CSV file containing the empirical source-redshift distribution. "
            f"Default: {DEFAULT_ZS_SAMPLES}"
        ),
    )
    parser.add_argument(
        "--zs-column",
        type=str,
        default=DEFAULT_ZS_COLUMN,
        help="CSV column used as the empirical redshift sample. Default: zgal",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for the source sampling. Default: 12345",
    )
    parser.add_argument(
        "--match-max-sep-arcsec",
        type=float,
        default=2.0,
        help="Maximum separation allowed when matching REF and RECON images. Default: 2.0",
    )
    parser.add_argument(
        "--caustic-buffer-arcsec",
        type=float,
        default=0.0,
        help="Optional buffer applied to the REF caustic union before sampling sources. Default: 0.0",
    )
    parser.add_argument(
        "--max-attempts-factor",
        type=int,
        default=40,
        help="Maximum sampling attempts per requested source = factor * N. Default: 40",
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
    parser.set_defaults(show=False)
    return parser.parse_args()


def resolve_output_path(output_path: Path, default_filename: str) -> Path:
    output_path = output_path.expanduser().resolve()
    if (output_path.exists() and output_path.is_dir()) or output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / default_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_source_catalog(source_records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_index",
        "z_source",
        "beta_1_arcsec",
        "beta_2_arcsec",
        "n_images_ref",
        "n_images_recon",
        "pos_rms_arcsec",
        "pos_frac_rms",
        "td_rel_rms_days",
        "td_rel_frac_rms",
        "td_rel_n_images",
        "td_rel_frac_n_images",
        "arrival_anchor_index",
        "sampled_from_shared_caustic",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, rec in enumerate(source_records, start=1):
            beta = rec.get("beta", (np.nan, np.nan))
            ref_x = np.asarray(rec.get("ref_x", []), dtype=float)
            recon_x = np.asarray(rec.get("recon_x", []), dtype=float)
            writer.writerow(
                {
                    "source_index": idx,
                    "z_source": float(rec.get("z", np.nan)),
                    "beta_1_arcsec": float(beta[0]) if len(beta) > 0 else float(np.nan),
                    "beta_2_arcsec": float(beta[1]) if len(beta) > 1 else float(np.nan),
                    "n_images_ref": int(ref_x.size),
                    "n_images_recon": int(recon_x.size),
                    "pos_rms_arcsec": float(rec.get("pos_rms_arcsec", np.nan)),
                    "pos_frac_rms": float(rec.get("pos_frac_rms", np.nan)),
                    "td_rel_rms_days": float(rec.get("td_rel_rms_days", np.nan)),
                    "td_rel_frac_rms": float(rec.get("td_rel_frac_rms", np.nan)),
                    "td_rel_n_images": int(np.asarray(rec.get("td_rel_diff", []), dtype=float).size),
                    "td_rel_frac_n_images": int(np.count_nonzero(np.isfinite(np.asarray(rec.get("td_rel_frac", []), dtype=float)))),
                    "arrival_anchor_index": int(rec.get("arrival_anchor_index", -1)),
                    "sampled_from_shared_caustic": bool(rec.get("sampled_from_shared_caustic", False)),
                }
            )


def write_image_catalog(
    source_records: list[dict[str, object]],
    output_path: Path,
    model_key: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_index",
        "z_source",
        "match_id",
        "image_index",
        "theta1_arcsec",
        "theta2_arcsec",
        "matched_theta1_arcsec",
        "matched_theta2_arcsec",
        "delta_theta1_recon_minus_ref_arcsec",
        "delta_theta2_recon_minus_ref_arcsec",
        "delta_theta_arcsec",
        "parity_eig1_sign",
        "parity_eig2_sign",
        "td_rel_days",
        "td_rel_diff_days",
        "td_rel_frac",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source_idx, rec in enumerate(source_records, start=1):
            xs = np.asarray(rec.get(f"{model_key}_x", []), dtype=float)
            ys = np.asarray(rec.get(f"{model_key}_y", []), dtype=float)
            other_key = "recon" if model_key == "ref" else "ref"
            other_xs = np.asarray(rec.get(f"{other_key}_x", []), dtype=float)
            other_ys = np.asarray(rec.get(f"{other_key}_y", []), dtype=float)
            ref_xs = np.asarray(rec.get("ref_x", []), dtype=float)
            ref_ys = np.asarray(rec.get("ref_y", []), dtype=float)
            recon_xs = np.asarray(rec.get("recon_x", []), dtype=float)
            recon_ys = np.asarray(rec.get("recon_y", []), dtype=float)
            sigs = np.asarray(rec.get(f"{model_key}_sig", []), dtype=int)
            tds = np.asarray(rec.get(f"{model_key}_td_rel", []), dtype=float)
            td_diff = np.asarray(rec.get("td_rel_diff", []), dtype=float)
            td_frac = np.asarray(rec.get("td_rel_frac", []), dtype=float)
            z_source = float(rec.get("z", np.nan))
            for image_idx, (x, y, td) in enumerate(zip(xs, ys, tds), start=1):
                offset = image_idx - 1
                other_x = float(other_xs[offset]) if offset < other_xs.size else float(np.nan)
                other_y = float(other_ys[offset]) if offset < other_ys.size else float(np.nan)
                dx = float(recon_xs[offset] - ref_xs[offset]) if offset < ref_xs.size and offset < recon_xs.size else float(np.nan)
                dy = float(recon_ys[offset] - ref_ys[offset]) if offset < ref_ys.size and offset < recon_ys.size else float(np.nan)
                td_diff_val = float(td_diff[image_idx - 1]) if image_idx - 1 < td_diff.size else float(np.nan)
                td_frac_val = float(td_frac[image_idx - 1]) if image_idx - 1 < td_frac.size else float(np.nan)
                writer.writerow(
                    {
                        "source_index": source_idx,
                        "z_source": z_source,
                        "match_id": image_idx,
                        "image_index": image_idx,
                        "theta1_arcsec": float(x),
                        "theta2_arcsec": float(y),
                        "matched_theta1_arcsec": other_x,
                        "matched_theta2_arcsec": other_y,
                        "delta_theta1_recon_minus_ref_arcsec": dx,
                        "delta_theta2_recon_minus_ref_arcsec": dy,
                        "delta_theta_arcsec": float(np.hypot(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else float(np.nan),
                        "parity_eig1_sign": int(sigs[offset, 0]) if sigs.ndim == 2 and offset < sigs.shape[0] else 0,
                        "parity_eig2_sign": int(sigs[offset, 1]) if sigs.ndim == 2 and offset < sigs.shape[0] else 0,
                        "td_rel_days": float(td),
                        "td_rel_diff_days": td_diff_val,
                        "td_rel_frac": td_frac_val,
                    }
                )


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
    if angx.ndim != 2 or angx.shape[0] != angx.shape[1]:
        raise ValueError(
            f"{prefix} maps must be square 2D arrays for this script; got shape {angx.shape}."
        )
    return angx, angy


def build_theta_grid(npix: int, pixel_scale_arcsec: float) -> np.ndarray:
    center = 0.5 * (npix - 1)
    return (np.arange(npix, dtype=float) - center) * pixel_scale_arcsec


def build_deflector_for_case(
    co: FlatLambdaCDM,
    angx: np.ndarray,
    angy: np.ndarray,
    zl: float,
    zs: float,
    pixel_scale_arcsec: float,
    compute_potential: bool = True,
) -> deflector:
    df = deflector(co, angx=angx, angy=angy, zl=zl, zs=zs)
    theta = build_theta_grid(angx.shape[0], pixel_scale_arcsec)
    df.setGrid(theta=theta, compute_potential=compute_potential)
    return df


def load_empirical_redshifts(csv_path: Path, preferred_column: str = "zgal") -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing redshift sample file: {csv_path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Redshift sample file has no header: {csv_path}")

        candidate_columns = [preferred_column]
        for fallback in ("zgal", "plane_z"):
            if fallback not in candidate_columns:
                candidate_columns.append(fallback)

        rows = list(reader)
        for column in candidate_columns:
            if column not in reader.fieldnames:
                continue
            values = []
            for row in rows:
                text = row.get(column, "")
                if text is None or text == "":
                    continue
                try:
                    value = float(text)
                except ValueError:
                    continue
                if np.isfinite(value):
                    values.append(value)
            if values:
                return np.asarray(values, dtype=float)

    raise ValueError(
        f"Could not find a usable redshift column in {csv_path}. "
        f"Tried: {preferred_column}, zgal, plane_z"
    )


def sample_empirical_redshifts(
    z_samples: np.ndarray,
    rng: np.random.Generator,
    n: int,
    zl: float,
) -> np.ndarray:
    z_samples = np.asarray(z_samples, dtype=float)
    z_samples = z_samples[np.isfinite(z_samples) & (z_samples > zl)]
    if z_samples.size == 0:
        raise ValueError("The empirical redshift sample has no values above the lens redshift.")
    return rng.choice(z_samples, size=int(n), replace=True)


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
        if geometry.contains(Point(x, y)):
            return x, y
    rp = geometry.representative_point()
    return float(rp.x), float(rp.y)


def caustic_union(df: deflector, buffer_size_arcsec: float = 0.0):
    clt = df.tancl()
    clr = df.radcl()
    caut = df.getCaustics(clt)
    caur = df.getCaustics(clr)
    caustics = list(caut) + list(caur)
    geoms = [c.geometria for c in caustics if not c.geometria.is_empty]
    if not geoms:
        return unary_union([])
    union = unary_union(geoms)
    if abs(buffer_size_arcsec) > 0.0:
        union = union.buffer(buffer_size_arcsec / df.pixel_scale)
    return union


def shared_caustic_region(
    df_ref: deflector,
    df_recon: deflector,
    buffer_size_arcsec: float = 0.0,
):
    """Return the intersection of the REF and RECON caustic regions."""
    ref_union = caustic_union(df_ref, buffer_size_arcsec=buffer_size_arcsec)
    recon_union = caustic_union(df_recon, buffer_size_arcsec=buffer_size_arcsec)
    if ref_union.is_empty or recon_union.is_empty:
        return unary_union([])
    shared = ref_union.intersection(recon_union)
    return shared


def source_arcsec_from_pixel(df: deflector, beta_pix: tuple[float, float]) -> tuple[float, float]:
    return (
        beta_pix[0] * df.pixel_scale + df.thetax[0],
        beta_pix[1] * df.pixel_scale + df.thetay[0],
    )


def solve_point_source(df: deflector, beta: tuple[float, float]) -> pointsrc:
    td_surface = df.t_delay_surf(beta=beta)
    return pointsrc(
        size=(df.thetax[-1] - df.thetax[0]),
        Npix=df.nray1,
        gl=df,
        ys1=beta[0],
        ys2=beta[1],
        zs=df.zs,
        refine=True,
        refine_to_td=True,
        td_surface=td_surface,
    )


def sample_source_inside_ref_caustics(
    df_ref: deflector,
    df_recon: deflector,
    rng: np.random.Generator,
    buffer_size_arcsec: float = 0.0,
    min_images: int = 2,
    max_tries: int = 200,
) -> tuple[tuple[float, float], pointsrc | None, pointsrc | None, bool]:
    """
    Sample a source inside the shared REF/RECON caustic region and require a
    consistent multiple-image solution in both models.
    """
    union = shared_caustic_region(df_ref, df_recon, buffer_size_arcsec=buffer_size_arcsec)
    fallback_pix = ((df_ref.nray1 - 1) / 2.0, (df_ref.nray2 - 1) / 2.0)
    if union.is_empty:
        beta = source_arcsec_from_pixel(df_ref, fallback_pix)
        return beta, None, None, False

    for _ in range(max_tries):
        beta_pix = sample_point_in_geometry(union, rng, fallback=fallback_pix)
        beta = source_arcsec_from_pixel(df_ref, beta_pix)
        ps_ref = solve_point_source(df_ref, beta)
        ps_recon = solve_point_source(df_recon, beta)
        if len(ps_ref.xi1) >= min_images and len(ps_ref.xi1) == len(ps_recon.xi1):
            return beta, ps_ref, ps_recon, True

    beta = source_arcsec_from_pixel(df_ref, fallback_pix)
    return beta, None, None, True


def interpolate_field(df: deflector, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ix = (np.asarray(x, dtype=float) - float(df.thetax[0])) / float(df.pixel_scale)
    iy = (np.asarray(y, dtype=float) - float(df.thetay[0])) / float(df.pixel_scale)
    coords = np.vstack([iy, ix])
    return np.asarray(map_coordinates(np.asarray(field, dtype=float), coords, order=1, mode="nearest"), dtype=float)


def relative_time_delays_for_images(
    df: deflector,
    beta: tuple[float, float],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute image arrival times and delays relative to the earliest image.
    """
    td = df.t_delay_surf(beta=beta)
    td_img = np.asarray(interpolate_field(df, td, x, y), dtype=float)
    finite = np.isfinite(td_img)
    if not np.any(finite):
        return td_img, td_img
    td_rel = td_img - float(np.nanmin(td_img[finite]))
    return td_img, td_rel


def parity_signature(df: deflector, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    kappa = interpolate_field(df, df.ka, x, y)
    g1 = interpolate_field(df, df.g1, x, y)
    g2 = interpolate_field(df, df.g2, x, y)
    sig = np.empty((len(np.atleast_1d(kappa)), 2), dtype=int)
    for i, (kk, gg1, gg2) in enumerate(zip(np.atleast_1d(kappa), np.atleast_1d(g1), np.atleast_1d(g2))):
        jac = np.array(
            [
                [1.0 - kk - gg1, -gg2],
                [-gg2, 1.0 - kk + gg1],
            ],
            dtype=float,
        )
        eigvals = np.linalg.eigvalsh(jac)
        sig[i, 0] = int(np.sign(eigvals[0]))
        sig[i, 1] = int(np.sign(eigvals[1]))
    return sig


def match_images_by_position_and_parity(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_sig: np.ndarray,
    recon_x: np.ndarray,
    recon_y: np.ndarray,
    recon_sig: np.ndarray,
    max_sep_arcsec: float,
) -> tuple[np.ndarray, np.ndarray]:
    ref_x = np.asarray(ref_x, dtype=float)
    ref_y = np.asarray(ref_y, dtype=float)
    recon_x = np.asarray(recon_x, dtype=float)
    recon_y = np.asarray(recon_y, dtype=float)
    ref_sig = np.asarray(ref_sig, dtype=int)
    recon_sig = np.asarray(recon_sig, dtype=int)

    if ref_x.size != recon_x.size:
        return np.array([], dtype=int), np.array([], dtype=int)

    if Counter(map(tuple, ref_sig)) != Counter(map(tuple, recon_sig)):
        return np.array([], dtype=int), np.array([], dtype=int)

    matched_ref: list[int] = []
    matched_recon: list[int] = []
    sigs = sorted(set(map(tuple, ref_sig)))
    for sig in sigs:
        ref_idx = np.where((ref_sig[:, 0] == sig[0]) & (ref_sig[:, 1] == sig[1]))[0]
        recon_idx = np.where((recon_sig[:, 0] == sig[0]) & (recon_sig[:, 1] == sig[1]))[0]
        if ref_idx.size != recon_idx.size:
            return np.array([], dtype=int), np.array([], dtype=int)
        if ref_idx.size == 0:
            continue
        ref_pos = np.column_stack([ref_x[ref_idx], ref_y[ref_idx]])
        recon_pos = np.column_stack([recon_x[recon_idx], recon_y[recon_idx]])
        dist = np.sqrt(((ref_pos[:, None, :] - recon_pos[None, :, :]) ** 2).sum(axis=2))
        row_ind, col_ind = linear_sum_assignment(dist)
        if np.any(dist[row_ind, col_ind] > max_sep_arcsec):
            return np.array([], dtype=int), np.array([], dtype=int)
        matched_ref.extend(ref_idx[row_ind].tolist())
        matched_recon.extend(recon_idx[col_ind].tolist())

    return np.asarray(matched_ref, dtype=int), np.asarray(matched_recon, dtype=int)


def match_source_images(
    df_ref: deflector,
    df_recon: deflector,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    recon_x: np.ndarray,
    recon_y: np.ndarray,
    max_sep_arcsec: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Match the REF and RECON images for one source using positions and Jacobian parity.

    Sources are rejected if the two models have different image numbers, if the parity
    signatures do not agree, or if any matched pair exceeds the allowed separation.
    """
    ref_x = np.asarray(ref_x, dtype=float)
    ref_y = np.asarray(ref_y, dtype=float)
    recon_x = np.asarray(recon_x, dtype=float)
    recon_y = np.asarray(recon_y, dtype=float)
    if ref_x.size != recon_x.size or ref_x.size == 0:
        return None

    ref_sig = parity_signature(df_ref, ref_x, ref_y)
    recon_sig = parity_signature(df_recon, recon_x, recon_y)
    if Counter(map(tuple, ref_sig)) != Counter(map(tuple, recon_sig)):
        return None

    row_sel, col_sel = match_images_by_position_and_parity(
        ref_x,
        ref_y,
        ref_sig,
        recon_x,
        recon_y,
        recon_sig,
        max_sep_arcsec=max_sep_arcsec,
    )
    if row_sel.size != ref_x.size:
        return None
    return row_sel, col_sel


def display_critical_lines(ax: plt.Axes, df: deflector) -> None:
    tancl = df.tancl()
    radcl = df.radcl()
    first = True
    for cl in tancl:
        x, y = df.getCritPoints(cl)
        ax.plot(x, y, color="tab:red", lw=1.3, alpha=0.95, label="Tangential critical line" if first else None)
        first = False
    first = True
    for cl in radcl:
        x, y = df.getCritPoints(cl)
        ax.plot(x, y, color="tab:blue", lw=1.0, ls="--", alpha=0.95, label="Radial critical line" if first else None)
        first = False


def plot_results(
    output_path: Path,
    show: bool,
    ref_display: deflector,
    source_records: list[dict[str, object]],
    matched_sticks: list[dict[str, object]],
    pos_frac_rms: float,
    td_rel_rms_days: float,
    td_rel_frac_rms: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.5), constrained_layout=False)
    ax_map, ax_disp = axes
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.08, top=0.90, wspace=0.16)

    kappa = np.asarray(ref_display.ka, dtype=float)
    theta_x = np.asarray(ref_display.thetax, dtype=float)
    theta_y = np.asarray(ref_display.thetay, dtype=float)
    vmin, vmax = np.nanpercentile(kappa[np.isfinite(kappa)], [2.0, 98.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(kappa))
        vmax = float(np.nanmax(kappa))
    if vmin == vmax:
        vmax = vmin + 1.0

    im = ax_map.imshow(
        kappa,
        origin="lower",
        extent=[theta_x[0], theta_x[-1], theta_y[0], theta_y[-1]],
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    display_critical_lines(ax_map, ref_display)

    src_colors = plt.cm.viridis
    if source_records:
        z_vals = np.array([float(rec["z"]) for rec in source_records], dtype=float)
        z_norm = Normalize(vmin=float(np.min(z_vals)), vmax=float(np.max(z_vals)) if np.max(z_vals) > np.min(z_vals) else float(np.min(z_vals) + 1.0))
    else:
        z_norm = Normalize(vmin=0.0, vmax=1.0)

    for rec in source_records:
        z = float(rec["z"])
        color = src_colors(z_norm(z))
        ref_x = np.asarray(rec["ref_x"], dtype=float)
        ref_y = np.asarray(rec["ref_y"], dtype=float)
        ax_map.scatter(
            ref_x,
            ref_y,
            s=18,
            color=color,
            edgecolors="white",
            linewidths=0.3,
            alpha=0.85,
            zorder=4,
        )

    ax_map.set_xlabel(r"$\theta_1$ [arcsec]")
    ax_map.set_ylabel(r"$\theta_2$ [arcsec]")
    ax_map.set_title("REF convergence map with critical lines and image positions", fontsize=12)
    ax_map.set_aspect("equal")
    ax_map.legend(frameon=False, loc="upper right", fontsize=9)
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\kappa$")

    if matched_sticks:
        all_ref_x = np.concatenate([np.asarray(rec["ref_x"], dtype=float) for rec in matched_sticks])
        all_ref_y = np.concatenate([np.asarray(rec["ref_y"], dtype=float) for rec in matched_sticks])
        all_recon_x = np.concatenate([np.asarray(rec["recon_x"], dtype=float) for rec in matched_sticks])
        all_recon_y = np.concatenate([np.asarray(rec["recon_y"], dtype=float) for rec in matched_sticks])
        delta_x = all_recon_x - all_ref_x
        delta_y = all_recon_y - all_ref_y
        all_displacements = np.sqrt(delta_x**2 + delta_y**2)
        span_x = max(
            float(all_ref_x.max() - all_ref_x.min()),
            float(all_recon_x.max() - all_recon_x.min()),
            1.0,
        )
        span_y = max(
            float(all_ref_y.max() - all_ref_y.min()),
            float(all_recon_y.max() - all_recon_y.min()),
            1.0,
        )
        pad_x = 0.15 * span_x
        pad_y = 0.15 * span_y
        x_min = float(np.min([all_ref_x.min(), all_recon_x.min()])) - pad_x
        x_max = float(np.max([all_ref_x.max(), all_recon_x.max()])) + pad_x
        y_min = float(np.min([all_ref_y.min(), all_recon_y.min()])) - pad_y
        y_max = float(np.max([all_ref_y.max(), all_recon_y.max()])) + pad_y
    else:
        all_ref_x = np.array([0.0])
        all_ref_y = np.array([0.0])
        all_recon_x = np.array([0.0])
        all_recon_y = np.array([0.0])
        delta_x = np.array([0.0])
        delta_y = np.array([0.0])
        all_displacements = np.array([0.0])
        x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0

    for rec in matched_sticks:
        z = float(rec["z"])
        color = src_colors(z_norm(z))
        ref_x = np.asarray(rec["ref_x"], dtype=float)
        ref_y = np.asarray(rec["ref_y"], dtype=float)
        recon_x = np.asarray(rec["recon_x"], dtype=float)
        recon_y = np.asarray(rec["recon_y"], dtype=float)
        for x0, y0, x1, y1 in zip(ref_x, ref_y, recon_x, recon_y):
            ax_disp.plot([x0, x1], [y0, y1], color=color, lw=2.1, alpha=0.65, zorder=1)

    ax_disp.axhline(0.0, color="0.85", lw=0.8)
    ax_disp.axvline(0.0, color="0.85", lw=0.8)
    ax_disp.set_xlim(float(theta_x[0]), float(theta_x[-1]))
    ax_disp.set_ylim(float(theta_y[0]), float(theta_y[-1]))
    ax_disp.set_aspect("equal", adjustable="box")
    if hasattr(ax_disp, "set_box_aspect"):
        ax_disp.set_box_aspect(1)
    ax_disp.set_xlabel(r"$\theta_1$ [arcsec]")
    ax_disp.set_ylabel(r"$\theta_2$ [arcsec]")
    ax_disp.set_title("Matched REF-to-RECON displacements", fontsize=12)

    rms_disp = float(np.sqrt(np.mean(all_displacements**2))) if all_displacements.size > 0 else float("nan")
    inset = inset_axes(ax_disp, width="42%", height="27%", loc="upper left", borderpad=2.2)
    bins = 18
    inset.hist(delta_x, bins=bins, color="tab:blue", alpha=0.55, edgecolor="white", label=r"$\Delta \theta_1$")
    inset.hist(delta_y, bins=bins, color="tab:orange", alpha=0.55, edgecolor="white", label=r"$\Delta \theta_2$")
    inset.axvline(0.0, color="black", ls="--", lw=1.0)
    inset.set_title("Signed displacements", fontsize=7, pad=2)
    inset.tick_params(labelsize=6)
    inset.set_xlabel(r"$\Delta \theta$ [arcsec]", fontsize=6, labelpad=1)
    inset.set_ylabel("N", fontsize=6, labelpad=1)
    inset.legend(frameon=False, fontsize=6, loc="best", handlelength=1.5)

    ax_disp.text(
        0.97,
        0.97,
        f"positional RMS = {rms_disp:.4f} arcsec\n"
        f"positional frac RMS = {pos_frac_rms:.4f}\n"
        f"TD rel RMS = {td_rel_rms_days:.4f} d\n"
        f"TD frac RMS = {td_rel_frac_rms:.4f}",
        transform=ax_disp.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3),
    )

    sm = plt.cm.ScalarMappable(norm=z_norm, cmap=src_colors)
    sm.set_array([])
    cbar2 = fig.colorbar(sm, ax=ax_disp, fraction=0.046, pad=0.02)
    cbar2.set_label(r"source redshift $z_s$")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    #fig.suptitle(
    #    "Source-redshift sampled image displacements: REF vs RECON",
    #    fontsize=14,
    #    y=0.97,
    #)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_td_histograms(
    output_path: Path,
    show: bool,
    td_rel_diffs_all: np.ndarray,
    td_rel_frac_all: np.ndarray,
    td_rel_rms_days: float,
    td_rel_frac_rms: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=False)
    ax_abs, ax_frac = axes
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.12, top=0.88, wspace=0.26)

    def _plot_hist(
        ax: plt.Axes,
        data: np.ndarray,
        title: str,
        xlabel: str,
        color: str,
        rms_text: str,
    ) -> None:
        finite = np.asarray(data, dtype=float)
        finite = finite[np.isfinite(finite)]
        ax.axvline(0.0, color="black", ls="--", lw=0.9)
        if finite.size > 0:
            clipped = finite
            if finite.size >= 8:
                lo, hi = np.percentile(finite, [1.0, 99.0])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo = float(np.nanmin(finite))
                    hi = float(np.nanmax(finite))
            else:
                lo = float(np.nanmin(finite))
                hi = float(np.nanmax(finite))
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            clipped = finite[(finite >= lo) & (finite <= hi)]
            if clipped.size == 0:
                clipped = finite
            bins = min(40, max(12, int(np.sqrt(clipped.size)) * 2))
            ax.hist(clipped, bins=bins, color=color, alpha=0.75, edgecolor="white")
            clipped_rms = float(np.sqrt(np.mean(clipped**2))) if clipped.size > 0 else float("nan")
            if clipped.size < finite.size:
                ax.text(
                    0.97,
                    0.05,
                    f"clipped {finite.size - clipped.size}\nclipped RMS = {clipped_rms:.4g}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.25),
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No finite data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of matched images")
        ax.text(
            0.97,
            0.97,
            f"full RMS = {rms_text}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
        )

    _plot_hist(
        ax_abs,
        td_rel_diffs_all,
        "Relative TD differences: RECON_rel - REF_rel",
        r"$\Delta t_\mathrm{rel}$ [d]",
        "tab:purple",
        f"{td_rel_rms_days:.4f} d",
    )
    _plot_hist(
        ax_frac,
        td_rel_frac_all,
        "Fractional relative TD differences",
        r"$\Delta t_\mathrm{rel} / t_{\mathrm{REF,rel}}$",
        "tab:green",
        f"{td_rel_frac_rms:.4f}",
    )

    fig.suptitle("Relative time-delay statistics", fontsize=14, y=0.95)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved TD histogram figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = resolve_output_path(args.output, DEFAULT_OUTPUT.name)
    output_dir = output_path.parent
    sources_csv_path = resolve_output_path(
        args.sources_csv if args.sources_csv is not None else output_dir / f"{output_path.stem}_sources.csv",
        f"{output_path.stem}_sources.csv",
    )
    ref_images_csv_path = output_dir / f"{output_path.stem}_ref_images.csv"
    recon_images_csv_path = output_dir / f"{output_path.stem}_recon_images.csv"
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    co = FlatLambdaCDM(H0=float(args.h0), Om0=float(args.om0))
    ref_ax, ref_ay = load_case(input_dir, "ref")
    recon_ax, recon_ay = load_case(input_dir, "recon")
    if ref_ax.shape != recon_ax.shape:
        raise ValueError(f"REF and RECON shapes differ: {ref_ax.shape} vs {recon_ax.shape}")

    zs_samples = load_empirical_redshifts(args.zs_samples_file.expanduser().resolve(), args.zs_column)
    rng = np.random.default_rng(int(args.seed))
    sampled_zs = sample_empirical_redshifts(zs_samples, rng, int(args.n_sources), float(args.zl))

    ref_display = build_deflector_for_case(
        co,
        ref_ax,
        ref_ay,
        float(args.zl),
        float(args.display_zs),
        float(args.pixel_scale_arcsec),
        compute_potential=False,
    )
    ref_display.change_redshift(float(args.display_zs))

    source_records: list[dict[str, object]] = []
    matched_sticks: list[dict[str, object]] = []
    td_rel_diffs_all: list[np.ndarray] = []
    td_rel_frac_all: list[np.ndarray] = []
    pos_frac_displacements_all: list[np.ndarray] = []
    accepted = 0
    total_matched_images = 0
    rejection_counts = Counter()

    for target_idx, z_target in enumerate(sampled_zs, start=1):
        z_target = float(z_target)

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
        df_ref.change_redshift(z_target)
        df_recon.change_redshift(z_target)

        beta, ref_ps, recon_ps, sampled_from_shared = sample_source_inside_ref_caustics(
            df_ref,
            df_recon,
            rng,
            buffer_size_arcsec=float(args.caustic_buffer_arcsec),
            min_images=2,
            max_tries=max(1, int(args.max_attempts_factor)),
        )
        if ref_ps is None or recon_ps is None:
            rejection_counts["no_consistent_multiple_images"] += 1
            if not sampled_from_shared:
                rejection_counts["empty_shared_caustic"] += 1
            continue

        ref_x = np.asarray(ref_ps.xi1, dtype=float)
        ref_y = np.asarray(ref_ps.xi2, dtype=float)
        recon_x = np.asarray(recon_ps.xi1, dtype=float)
        recon_y = np.asarray(recon_ps.xi2, dtype=float)
        if ref_x.size != recon_x.size or ref_x.size == 0:
            rejection_counts["image_count_mismatch"] += 1
            continue

        matched = match_source_images(
            df_ref,
            df_recon,
            ref_x,
            ref_y,
            recon_x,
            recon_y,
            max_sep_arcsec=float(args.match_max_sep_arcsec),
        )
        if matched is None:
            rejection_counts["position_parity_match_failure"] += 1
            continue
        row_sel, col_sel = matched

        ref_x = ref_x[row_sel]
        ref_y = ref_y[row_sel]
        recon_x = recon_x[col_sel]
        recon_y = recon_y[col_sel]
        ref_sig = parity_signature(df_ref, ref_x, ref_y)
        recon_sig = parity_signature(df_recon, recon_x, recon_y)
        ref_td_raw, ref_td_rel = relative_time_delays_for_images(df_ref, beta, ref_x, ref_y)
        recon_td_raw, recon_td_rel = relative_time_delays_for_images(df_recon, beta, recon_x, recon_y)
        if ref_td_rel.size != recon_td_rel.size or ref_td_rel.size != ref_x.size:
            rejection_counts["td_interpolation_failure"] += 1
            continue
        if not np.all(np.isfinite(ref_td_raw)) or not np.all(np.isfinite(recon_td_raw)):
            rejection_counts["td_interpolation_failure"] += 1
            continue
        ref_anchor_idx = int(np.nanargmin(ref_td_raw))
        recon_anchor_idx = int(np.nanargmin(recon_td_raw))
        if ref_anchor_idx != recon_anchor_idx:
            rejection_counts["arrival_anchor_mismatch"] += 1
            continue
        td_rel_diff = recon_td_rel - ref_td_rel
        with np.errstate(divide="ignore", invalid="ignore"):
            td_rel_frac = np.divide(
                td_rel_diff,
                ref_td_rel,
                out=np.full_like(td_rel_diff, np.nan, dtype=float),
                where=np.abs(ref_td_rel) > np.finfo(float).eps,
            )
        td_rel_rms = float(np.sqrt(np.mean(td_rel_diff**2))) if td_rel_diff.size > 0 else float("nan")
        td_rel_frac_valid = td_rel_frac[np.isfinite(td_rel_frac)]
        td_rel_frac_rms = float(np.sqrt(np.mean(td_rel_frac_valid**2))) if td_rel_frac_valid.size > 0 else float("nan")
        pos_disp = np.sqrt((recon_x - ref_x) ** 2 + (recon_y - ref_y) ** 2)
        ref_radius = np.sqrt(ref_x**2 + ref_y**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            pos_frac = np.divide(
                pos_disp,
                ref_radius,
                out=np.full_like(pos_disp, np.nan, dtype=float),
                where=ref_radius > np.finfo(float).eps,
            )
        pos_frac_valid = pos_frac[np.isfinite(pos_frac)]
        pos_frac_rms = float(np.sqrt(np.mean(pos_frac_valid**2))) if pos_frac_valid.size > 0 else float("nan")

        source_records.append(
            {
                "z": float(z_target),
                "beta": beta,
                "ref_x": ref_x,
                "ref_y": ref_y,
                "recon_x": recon_x,
                "recon_y": recon_y,
                "ref_sig": ref_sig,
                "recon_sig": recon_sig,
                "ref_td_rel": ref_td_rel,
                "recon_td_rel": recon_td_rel,
                "td_rel_diff": td_rel_diff,
                "td_rel_frac": td_rel_frac,
                "pos_rms_arcsec": float(np.sqrt(np.mean(pos_disp**2))) if pos_disp.size > 0 else float("nan"),
                "pos_frac_rms": pos_frac_rms,
                "td_rel_rms_days": td_rel_rms,
                "td_rel_frac_rms": td_rel_frac_rms,
                "arrival_anchor_index": ref_anchor_idx + 1,
                "sampled_from_shared_caustic": sampled_from_shared,
                "distribution_file": str(args.zs_samples_file.expanduser().resolve()),
            }
        )
        td_rel_diffs_all.append(td_rel_diff)
        if td_rel_frac_valid.size > 0:
            td_rel_frac_all.append(td_rel_frac_valid)
        if pos_frac_valid.size > 0:
            pos_frac_displacements_all.append(pos_frac_valid)
        matched_sticks.append(
            {
                "z": float(z_target),
                "beta": beta,
                "ref_x": ref_x,
                "ref_y": ref_y,
                "recon_x": recon_x,
                "recon_y": recon_y,
            }
        )
        total_matched_images += int(ref_x.size)
        accepted += 1
        print(
            f"[{accepted}/{args.n_sources}] z={z_target:.3f}: "
            f"matched images={len(ref_x)}, "
            f"A={ref_anchor_idx + 1}, "
            f"pos frac RMS={pos_frac_rms:.4f}, "
            f"TD rel RMS={td_rel_rms:.4f} d, frac RMS={td_rel_frac_rms:.4f}"
        )

        if target_idx % 25 == 0:
            print(f"Progress: sampled redshifts={target_idx}, accepted={accepted}")

    if not source_records:
        raise RuntimeError("No accepted sources were generated.")

    if accepted < int(args.n_sources):
        print(
            f"Warning: only accepted {accepted} sources from {int(args.n_sources)} sampled redshifts. "
            f"Requested {int(args.n_sources)}."
        )

    print(f"Total matched images: {total_matched_images}")
    if rejection_counts:
        print("Rejected sampled redshifts:")
        for reason, count in sorted(rejection_counts.items()):
            print(f"  {reason}: {count}")
    all_td_rel_diffs = np.concatenate(td_rel_diffs_all) if td_rel_diffs_all else np.array([], dtype=float)
    all_td_rel_frac = np.concatenate(td_rel_frac_all) if td_rel_frac_all else np.array([], dtype=float)
    td_rel_rms_global = float(np.sqrt(np.mean(all_td_rel_diffs**2))) if all_td_rel_diffs.size > 0 else float("nan")
    td_rel_frac_rms_global = float(np.sqrt(np.mean(all_td_rel_frac**2))) if all_td_rel_frac.size > 0 else float("nan")
    all_pos_frac_displacements = np.concatenate(pos_frac_displacements_all) if pos_frac_displacements_all else np.array([], dtype=float)
    pos_frac_rms_global = float(np.sqrt(np.mean(all_pos_frac_displacements**2))) if all_pos_frac_displacements.size > 0 else float("nan")
    print(f"Fractional positional RMS: {pos_frac_rms_global:.6f}")
    print(f"Relative TD RMS (RECON_rel - REF_rel): {td_rel_rms_global:.6f} d")
    print(f"Fractional relative TD RMS: {td_rel_frac_rms_global:.6f}")

    write_source_catalog(source_records, sources_csv_path)
    print(f"Saved source catalog to {sources_csv_path}")
    write_image_catalog(source_records, ref_images_csv_path, "ref")
    print(f"Saved REF image catalog to {ref_images_csv_path}")
    write_image_catalog(source_records, recon_images_csv_path, "recon")
    print(f"Saved RECON image catalog to {recon_images_csv_path}")

    plot_results(
        output_path=output_path,
        show=bool(args.show),
        ref_display=ref_display,
        source_records=source_records,
        matched_sticks=matched_sticks,
        pos_frac_rms=pos_frac_rms_global,
        td_rel_rms_days=td_rel_rms_global,
        td_rel_frac_rms=td_rel_frac_rms_global,
    )

    td_hist_output = output_path.with_name(f"{output_path.stem}_td_histograms{output_path.suffix}")
    plot_td_histograms(
        output_path=td_hist_output,
        show=bool(args.show),
        td_rel_diffs_all=all_td_rel_diffs,
        td_rel_frac_all=all_td_rel_frac,
        td_rel_rms_days=td_rel_rms_global,
        td_rel_frac_rms=td_rel_frac_rms_global,
    )


if __name__ == "__main__":
    main()
