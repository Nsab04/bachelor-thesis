#!/usr/bin/env python3
"""
Compare lensing maps for REF and RECON deflection outputs.

The script produces four separate figures:
  1. deflection angle components
  2. convergence
  3. shear components
  4. magnification on the lens and source planes

Each figure also includes the residual map (RECON-REF)/REF.

The source-plane magnification is computed with pymupds and requires the
compiled extension to be available at runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import numpy as np
from astropy.cosmology import FlatLambdaCDM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_DIR = Path("/Users/maxmen3/Downloads/deflection_outputs")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "compare_lensing_maps.png"

from pyLensLib.deflector import deflector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare lensing maps for REF and RECON deflection outputs."
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
        "--zl",
        type=float,
        required=True,
        help="Lens redshift used to build the deflector.",
    )
    parser.add_argument(
        "--zs",
        type=float,
        default=1.6,
        help="Source redshift used to normalize the deflection maps. Default: 1.6",
    )
    parser.add_argument(
        "--pixel-scale-arcsec",
        type=float,
        required=True,
        help="Pixel scale of the deflection maps in arcsec/pixel.",
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
    zs: float,
    pixel_scale_arcsec: float,
) -> deflector:
    df = deflector(co, angx=angx, angy=angy, zl=zl, zs=zs)
    theta = build_theta_grid(angx.shape[0], pixel_scale_arcsec)
    df.setGrid(theta=theta, compute_potential=False)
    return df


def finite_values(*arrays: np.ndarray) -> np.ndarray:
    vals = [np.asarray(arr, dtype=float).ravel() for arr in arrays]
    if not vals:
        return np.array([], dtype=float)
    stacked = np.concatenate(vals)
    return stacked[np.isfinite(stacked)]


def robust_linear_norm(*arrays: np.ndarray, symmetric: bool = False) -> Normalize:
    vals = finite_values(*arrays)
    if vals.size == 0:
        return Normalize(vmin=-1.0, vmax=1.0)
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(vals), 98.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.nanmax(np.abs(vals))) if np.any(np.isfinite(vals)) else 1.0
        vmax = max(vmax, 1e-12)
        return Normalize(vmin=-vmax, vmax=vmax)
    vmin, vmax = np.nanpercentile(vals, [2.0, 98.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vals)) if np.any(np.isfinite(vals)) else -1.0
        vmax = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else 1.0
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def robust_symlog_norm(*arrays: np.ndarray) -> SymLogNorm:
    vals = finite_values(*arrays)
    if vals.size == 0:
        return SymLogNorm(linthresh=1.0, vmin=-1.0, vmax=1.0)
    vmax = float(np.nanpercentile(np.abs(vals), 98.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(np.abs(vals))) if np.any(np.isfinite(vals)) else 1.0
    vmax = max(vmax, 1e-12)
    linthresh = float(np.nanpercentile(np.abs(vals), 5.0))
    if not np.isfinite(linthresh) or linthresh <= 0.0:
        linthresh = vmax * 1e-2
    linthresh = max(linthresh, vmax * 1e-4, 1e-12)
    return SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax, vmax=vmax)


def robust_lognorm(*arrays: np.ndarray) -> LogNorm:
    vals = finite_values(*arrays)
    vals = vals[vals > 0.0]
    if vals.size == 0:
        return LogNorm(vmin=1e-3, vmax=1.0)
    vmin, vmax = np.nanpercentile(vals, [2.0, 98.0])
    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = float(np.nanmin(vals))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(vals))
    if vmin == vmax:
        vmax = vmin * 10.0
    return LogNorm(vmin=max(vmin, np.finfo(float).tiny), vmax=max(vmax, vmin * 10.0))


def compute_lens_plane_maps(df: deflector) -> tuple[np.ndarray, np.ndarray]:
    detA = (1.0 - df.ka) ** 2 - (df.g1 ** 2 + df.g2 ** 2)
    mu_lens = np.divide(
        1.0,
        detA,
        out=np.full_like(detA, np.nan, dtype=float),
        where=np.abs(detA) > 1e-12,
    )
    return detA, mu_lens


def relative_residual(ref: np.ndarray, recon: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=float)
    recon = np.asarray(recon, dtype=float)
    return np.divide(
        recon - ref,
        ref,
        out=np.full_like(ref, np.nan, dtype=float),
        where=np.abs(ref) > np.finfo(float).eps,
    )


def import_pymupds():
    try:
        import pymupds  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pymupds is required for source-plane magnification but was not found. "
            "Please install or compile pymupds and make sure it is importable."
        ) from exc
    return pymupds


def compute_source_plane_magnification(
    df: deflector,
    detA: np.ndarray,
    pymupds_mod,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The pymupds convention expects the source-plane coordinates shifted to
    # start at zero, matching the examples and tests bundled with pymupds.
    source_x = df.theta1 - df.a1 - float(df.thetax.min())
    source_y = df.theta2 - df.a2 - float(df.thetay.min())

    if hasattr(pymupds_mod, "mupds_triangle"):
        mus = np.zeros_like(detA, dtype=np.float64, order="F")
        xray = np.asfortranarray(source_x, dtype=np.float64)
        yray = np.asfortranarray(source_y, dtype=np.float64)
        deta = np.asfortranarray(detA, dtype=np.float64)
        pymupds_mod.mupds_triangle(mus, int(df.nray1), xray, yray, float(df.pixel_scale), deta)
        mus = np.asarray(mus, dtype=float)
    else:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "pymupds is importable, but it does not expose mupds_triangle(). "
            "Please rebuild/reinstall the compiled extension."
        )

    mus = np.asarray(mus, dtype=float)
    mus[mus <= 0.0] = np.nan
    return mus, source_x, source_y


def make_output_path(output_path: Path, suffix: str) -> Path:
    if (output_path.exists() and output_path.is_dir()) or output_path.suffix == "":
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"compare_lensing_maps_{suffix}.png"
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def plot_triptych_figure(
    *,
    figure_title: str,
    row_specs: list[dict[str, object]],
    output_path: Path,
    show: bool,
) -> None:
    nrows = len(row_specs)
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 4.8 * nrows), constrained_layout=False)
    if nrows == 1:
        axes = np.asarray([axes])
    top_margin = 0.84 if nrows == 1 else 0.90
    title_y = 0.955 if nrows == 1 else 0.965
    title_pad = 12 if nrows == 1 else 10
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.08, top=top_margin, wspace=0.18, hspace=0.14)
    fig.suptitle(figure_title, fontsize=14, y=title_y)

    col_titles = ["REF", "RECON", "(RECON-REF)/REF"]
    for row_idx, spec in enumerate(row_specs):
        row_label = str(spec["label"])
        ref_data = np.asarray(spec["ref"], dtype=float)
        recon_data = np.asarray(spec["recon"], dtype=float)
        resid_data = np.asarray(spec["resid"], dtype=float)
        ref_norm = spec["ref_norm"]
        resid_norm = spec["resid_norm"]
        ref_cmap = str(spec["ref_cmap"])
        resid_cmap = str(spec["resid_cmap"])
        extent_ref = spec["ref_extent"]
        extent_recon = spec["recon_extent"]
        extent_resid = spec["resid_extent"]

        panel_data = [
            (ref_data, extent_ref, ref_cmap, ref_norm),
            (recon_data, extent_recon, ref_cmap, ref_norm),
            (resid_data, extent_resid, resid_cmap, resid_norm),
        ]
        for col_idx, (data, extent, cmap, norm) in enumerate(panel_data):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=norm,
                aspect="equal",
            )
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, pad=title_pad)
            if col_idx == 0:
                ax.set_ylabel(row_label, rotation=0, labelpad=72, va="center", fontsize=11)
            else:
                ax.set_ylabel("")
            if row_idx == nrows - 1:
                ax.set_xlabel(r"$\theta_1$ [arcsec]")
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelsize=8)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved comparison figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    ref_ax, ref_ay = load_case(input_dir, "ref")
    recon_ax, recon_ay = load_case(input_dir, "recon")
    if ref_ax.shape != recon_ax.shape:
        raise ValueError(f"REF and RECON shapes differ: {ref_ax.shape} vs {recon_ax.shape}")

    co = FlatLambdaCDM(H0=float(args.h0), Om0=float(args.om0))
    df_ref = build_deflector_for_case(co, ref_ax, ref_ay, float(args.zl), float(args.zs), float(args.pixel_scale_arcsec))
    df_recon = build_deflector_for_case(co, recon_ax, recon_ay, float(args.zl), float(args.zs), float(args.pixel_scale_arcsec))

    detA_ref, mu_lens_ref = compute_lens_plane_maps(df_ref)
    detA_recon, mu_lens_recon = compute_lens_plane_maps(df_recon)

    pymupds_mod = import_pymupds()
    source_mu_ref, source_x_ref, source_y_ref = compute_source_plane_magnification(df_ref, detA_ref, pymupds_mod)
    source_mu_recon, source_x_recon, source_y_recon = compute_source_plane_magnification(df_recon, detA_recon, pymupds_mod)

    source_extent_ref = (
        float(np.nanmin(source_x_ref)),
        float(np.nanmax(source_x_ref)),
        float(np.nanmin(source_y_ref)),
        float(np.nanmax(source_y_ref)),
    )
    source_extent_recon = (
        float(np.nanmin(source_x_recon)),
        float(np.nanmax(source_x_recon)),
        float(np.nanmin(source_y_recon)),
        float(np.nanmax(source_y_recon)),
    )

    deflection_spec = [
        {
            "label": r"Deflection $\alpha_x$ [arcsec]",
            "ref": df_ref.a1,
            "recon": df_recon.a1,
            "resid": relative_residual(df_ref.a1, df_recon.a1),
            "ref_norm": robust_symlog_norm(df_ref.a1, df_recon.a1),
            "resid_norm": robust_symlog_norm(relative_residual(df_ref.a1, df_recon.a1)),
            "ref_cmap": "RdBu_r",
            "resid_cmap": "RdBu_r",
            "ref_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
            "recon_extent": [float(df_recon.thetax.min()), float(df_recon.thetax.max()), float(df_recon.thetay.min()), float(df_recon.thetay.max())],
            "resid_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
        },
        {
            "label": r"Deflection $\alpha_y$ [arcsec]",
            "ref": df_ref.a2,
            "recon": df_recon.a2,
            "resid": relative_residual(df_ref.a2, df_recon.a2),
            "ref_norm": robust_symlog_norm(df_ref.a2, df_recon.a2),
            "resid_norm": robust_symlog_norm(relative_residual(df_ref.a2, df_recon.a2)),
            "ref_cmap": "RdBu_r",
            "resid_cmap": "RdBu_r",
            "ref_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
            "recon_extent": [float(df_recon.thetax.min()), float(df_recon.thetax.max()), float(df_recon.thetay.min()), float(df_recon.thetay.max())],
            "resid_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
        },
    ]

    convergence_spec = [
        {
            "label": r"Convergence $\kappa$",
            "ref": df_ref.ka,
            "recon": df_recon.ka,
            "resid": relative_residual(df_ref.ka, df_recon.ka),
            "ref_norm": robust_linear_norm(df_ref.ka, df_recon.ka),
            "resid_norm": robust_symlog_norm(relative_residual(df_ref.ka, df_recon.ka)),
            "ref_cmap": "viridis",
            "resid_cmap": "RdBu_r",
            "ref_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
            "recon_extent": [float(df_recon.thetax.min()), float(df_recon.thetax.max()), float(df_recon.thetay.min()), float(df_recon.thetay.max())],
            "resid_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
        }
    ]

    shear_spec = [
        {
            "label": r"Shear $\gamma_1$",
            "ref": df_ref.g1,
            "recon": df_recon.g1,
            "resid": relative_residual(df_ref.g1, df_recon.g1),
            "ref_norm": robust_symlog_norm(df_ref.g1, df_recon.g1),
            "resid_norm": robust_symlog_norm(relative_residual(df_ref.g1, df_recon.g1)),
            "ref_cmap": "RdBu_r",
            "resid_cmap": "RdBu_r",
            "ref_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
            "recon_extent": [float(df_recon.thetax.min()), float(df_recon.thetax.max()), float(df_recon.thetay.min()), float(df_recon.thetay.max())],
            "resid_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
        },
        {
            "label": r"Shear $\gamma_2$",
            "ref": df_ref.g2,
            "recon": df_recon.g2,
            "resid": relative_residual(df_ref.g2, df_recon.g2),
            "ref_norm": robust_symlog_norm(df_ref.g2, df_recon.g2),
            "resid_norm": robust_symlog_norm(relative_residual(df_ref.g2, df_recon.g2)),
            "ref_cmap": "RdBu_r",
            "resid_cmap": "RdBu_r",
            "ref_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
            "recon_extent": [float(df_recon.thetax.min()), float(df_recon.thetax.max()), float(df_recon.thetay.min()), float(df_recon.thetay.max())],
            "resid_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
        },
    ]

    magnification_spec = [
        {
            "label": r"Lens-plane $\mu$",
            "ref": mu_lens_ref,
            "recon": mu_lens_recon,
            "resid": relative_residual(mu_lens_ref, mu_lens_recon),
            "ref_norm": robust_symlog_norm(mu_lens_ref, mu_lens_recon),
            "resid_norm": robust_symlog_norm(relative_residual(mu_lens_ref, mu_lens_recon)),
            "ref_cmap": "RdBu_r",
            "resid_cmap": "RdBu_r",
            "ref_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
            "recon_extent": [float(df_recon.thetax.min()), float(df_recon.thetax.max()), float(df_recon.thetay.min()), float(df_recon.thetay.max())],
            "resid_extent": [float(df_ref.thetax.min()), float(df_ref.thetax.max()), float(df_ref.thetay.min()), float(df_ref.thetay.max())],
        },
        {
            "label": r"Source-plane $\mu$",
            "ref": source_mu_ref.T,
            "recon": source_mu_recon.T,
            "resid": relative_residual(source_mu_ref.T, source_mu_recon.T),
            "ref_norm": robust_lognorm(source_mu_ref, source_mu_recon),
            "resid_norm": robust_symlog_norm(relative_residual(source_mu_ref.T, source_mu_recon.T)),
            "ref_cmap": "magma",
            "resid_cmap": "RdBu_r",
            "ref_extent": source_extent_ref,
            "recon_extent": source_extent_recon,
            "resid_extent": source_extent_ref,
        },
    ]

    plot_triptych_figure(
        figure_title=f"Deflection angle components (zl={df_ref.zl:.2f}, zs={df_ref.zs:.2f})",
        row_specs=deflection_spec,
        output_path=make_output_path(output_path, "deflection_components"),
        show=bool(args.show),
    )
    plot_triptych_figure(
        figure_title=f"Convergence (zl={df_ref.zl:.2f}, zs={df_ref.zs:.2f})",
        row_specs=convergence_spec,
        output_path=make_output_path(output_path, "convergence"),
        show=bool(args.show),
    )
    plot_triptych_figure(
        figure_title=f"Shear components (zl={df_ref.zl:.2f}, zs={df_ref.zs:.2f})",
        row_specs=shear_spec,
        output_path=make_output_path(output_path, "shear_components"),
        show=bool(args.show),
    )
    plot_triptych_figure(
        figure_title=f"Magnification maps (zl={df_ref.zl:.2f}, zs={df_ref.zs:.2f})",
        row_specs=magnification_spec,
        output_path=make_output_path(output_path, "magnification"),
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
