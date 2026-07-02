#!/usr/bin/env python3
"""Validate the point-source image finder against an analytic SIS lens.

This script builds a synthetic singular isothermal sphere (SIS) lens with an
analytic lens equation and analytically known image positions. It then compares
the raw image finder output and the refined output produced by ``pointsrc`` for
100 random sources sampled inside the SIS multiple-imaging region.

The workflow is:
1. Build an analytic SIS lens on a regular grid.
2. Sample source positions inside the Einstein radius.
3. Compute analytic image positions.
4. Compute numerical image positions with and without refinement.
5. Match the numerical and analytic images by Jacobian-sign signature and position.
6. Summarize the positional residuals and save a comparison figure plus CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from pyLensLib.deflector import deflector
from pyLensLib.pointsrc import pointsrc


DEFAULT_OUTPUT = Path("validate_image_finder_sis.png")


def resolve_output_paths(output_arg: str | None) -> tuple[Path, Path]:
    """Return figure and CSV output paths from a file or directory argument."""
    if output_arg is None:
        output = DEFAULT_OUTPUT
    else:
        output = Path(output_arg)

    if output.suffix:
        fig_path = output
        out_dir = output.parent
        stem = output.stem
    else:
        out_dir = output
        stem = "validate_image_finder_sis"
        fig_path = out_dir / f"{stem}.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    return fig_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the image finder and refinement against an analytic SIS lens."
        )
    )
    parser.add_argument(
        "--n-sources",
        type=int,
        default=100,
        help="Number of accepted sources to test.",
    )
    parser.add_argument(
        "--zl",
        type=float,
        default=0.5,
        help="Lens redshift used for the synthetic deflector.",
    )
    parser.add_argument(
        "--zs",
        type=float,
        default=2.0,
        help="Source redshift used for the synthetic deflector and source sampling.",
    )
    parser.add_argument(
        "--theta-e",
        type=float,
        default=1.5,
        help="Einstein radius of the SIS lens in arcsec.",
    )
    parser.add_argument(
        "--npix",
        type=int,
        default=1001,
        help="Number of pixels along each axis for the synthetic lens grid.",
    )
    parser.add_argument(
        "--fov-factor",
        type=float,
        default=6.0,
        help="Field of view in units of theta_E.",
    )
    parser.add_argument(
        "--source-rmin-fraction",
        type=float,
        default=0.05,
        help="Minimum source radius as a fraction of theta_E.",
    )
    parser.add_argument(
        "--source-rmax-fraction",
        type=float,
        default=0.85,
        help="Maximum source radius as a fraction of theta_E.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for source sampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=(
            "Output figure path or directory. If a directory is given, the script "
            "writes validate_image_finder_sis.png and a companion CSV inside it."
        ),
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display the figure interactively.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not display the figure.",
    )
    parser.set_defaults(show=False)
    return parser.parse_args()


def build_analytic_sis_deflector(
    theta_e: float,
    zl: float,
    zs: float,
    npix: int,
    fov_factor: float,
) -> tuple[deflector, np.ndarray]:
    """Construct a synthetic SIS deflector with an analytic potential map."""
    co = FlatLambdaCDM(H0=70.0, Om0=0.3)
    fov = fov_factor * theta_e
    theta = np.linspace(-0.5 * fov, 0.5 * fov, npix)
    theta1, theta2 = np.meshgrid(theta, theta)

    # SIS lensing potential psi(theta) = theta_E * |theta|.
    # The center pixel is regularized only by the finite grid sampling.
    pot = theta_e * np.hypot(theta1, theta2)

    df = deflector(co, pot=pot, usePotential=True, zl=zl, zs=zs)
    df.setGrid(theta=theta, compute_potential=False)
    return df, theta


def sample_source_position(
    rng: np.random.Generator,
    theta_e: float,
    rmin_fraction: float,
    rmax_fraction: float,
) -> tuple[float, float, float]:
    """Sample a source position inside the SIS multiple-imaging region."""
    rmin = max(0.0, rmin_fraction * theta_e)
    rmax = min(0.999 * theta_e, rmax_fraction * theta_e)
    if rmax <= rmin:
        raise ValueError("source-rmax-fraction must be larger than source-rmin-fraction.")

    radius = np.sqrt(rng.uniform(rmin**2, rmax**2))
    angle = rng.uniform(0.0, 2.0 * np.pi)
    beta1 = radius * np.cos(angle)
    beta2 = radius * np.sin(angle)
    return beta1, beta2, radius


def analytic_sis_images(
    beta1: float,
    beta2: float,
    theta_e: float,
) -> list[dict[str, float]]:
    """Return the analytic SIS image positions and signed magnifications."""
    beta = float(np.hypot(beta1, beta2))
    if beta <= 0.0 or beta >= theta_e:
        return []

    ux = beta1 / beta
    uy = beta2 / beta

    r_out = beta + theta_e
    r_in = theta_e - beta

    images = [
        {
            "parity": +1,
            "x": r_out * ux,
            "y": r_out * uy,
            "mu": 1.0 + theta_e / beta,
        },
        {
            "parity": -1,
            "x": -r_in * ux,
            "y": -r_in * uy,
            "mu": 1.0 - theta_e / beta,
        },
    ]

    return images


def sample_field(df: deflector, field: np.ndarray, x_arcsec: float, y_arcsec: float) -> float:
    """Sample a 2D lens field at an arcsec position using bilinear interpolation."""
    from scipy.ndimage import map_coordinates

    xpix = float(x_arcsec) / df.pixel_scale + (df.nray1 - 1) / 2.0
    ypix = float(y_arcsec) / df.pixel_scale + (df.nray2 - 1) / 2.0
    return float(map_coordinates(field, [[ypix], [xpix]], order=1, mode="nearest")[0])


def jacobian_signature(df: deflector, x_arcsec: float, y_arcsec: float) -> tuple[int, int]:
    """Return the signs of the ordered Jacobian eigenvalues at a position."""
    kappa = sample_field(df, df.ka, x_arcsec, y_arcsec)
    g1 = sample_field(df, df.g1, x_arcsec, y_arcsec)
    g2 = sample_field(df, df.g2, x_arcsec, y_arcsec)
    jac = np.array(
        [
            [1.0 - kappa - g1, -g2],
            [-g2, 1.0 - kappa + g1],
        ],
        dtype=float,
    )
    eigvals = np.linalg.eigvalsh(jac)
    return tuple(int(np.sign(val)) for val in eigvals)


def compute_numerical_images(
    df: deflector,
    beta1: float,
    beta2: float,
    npix: int,
    fov: float,
    zs: float,
    refine: bool,
) -> pointsrc:
    """Run the numerical image finder with optional refinement."""
    return pointsrc(
        size=fov,
        Npix=npix,
        gl=df,
        ys1=beta1,
        ys2=beta2,
        zs=zs,
        refine=refine,
        refine_to_td=refine,
    )


def match_images_by_signature(
    analytic_images: list[dict[str, float]],
    numeric_images: list[dict[str, float]],
) -> list[tuple[dict[str, float], dict[str, float]]]:
    """Match analytic and numerical images by Jacobian-sign signature."""
    matched: list[tuple[dict[str, float], dict[str, float]]] = []

    for ana in analytic_images:
        signature = tuple(int(v) for v in ana["signature"])
        candidates = [num for num in numeric_images if tuple(int(v) for v in num["signature"]) == signature]
        if not candidates:
            return []
        num = min(
            candidates,
            key=lambda img: float((img["x"] - ana["x"]) ** 2 + (img["y"] - ana["y"]) ** 2),
        )
        matched.append((ana, num))

    return matched


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {"mean": np.nan, "median": np.nan, "rms": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "rms": float(np.sqrt(np.mean(errors**2))),
        "max": float(np.max(errors)),
    }


def write_catalog(rows: list[dict[str, float]], csv_path: Path) -> None:
    fieldnames = [
        "source_index",
        "beta1_arcsec",
        "beta2_arcsec",
        "beta_radius_arcsec",
        "beta_angle_rad",
        "n_images_raw",
        "n_images_refined",
        "n_extra_images_raw",
        "n_extra_images_refined",
        "signature_1",
        "signature_2",
        "analytic_x_arcsec",
        "analytic_y_arcsec",
        "analytic_mu",
        "raw_x_arcsec",
        "raw_y_arcsec",
        "raw_mu",
        "refined_x_arcsec",
        "refined_y_arcsec",
        "refined_mu",
        "raw_dx_arcsec",
        "raw_dy_arcsec",
        "raw_dr_arcsec",
        "refined_dx_arcsec",
        "refined_dy_arcsec",
        "refined_dr_arcsec",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(
    output_path: Path,
    raw_errors: np.ndarray,
    refined_errors: np.ndarray,
) -> None:
    """Plot histograms and raw-vs-refined residual comparison."""
    fig = plt.figure(figsize=(13.5, 5.5))
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax_hist = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])

    all_errors = np.concatenate([raw_errors, refined_errors]) if raw_errors.size or refined_errors.size else np.array([])
    positive_errors = all_errors[all_errors > 0.0]
    err_max = float(np.max(all_errors)) if all_errors.size else 1.0
    if positive_errors.size:
        floor = max(1e-10, 0.5 * float(np.min(positive_errors)))
    else:
        floor = 1e-10
    top = max(err_max * 1.05, floor * 10.0)
    bins = np.logspace(np.log10(floor), np.log10(top), 36)

    ax_hist.hist(
        raw_errors,
        bins=bins,
        alpha=0.6,
        color="#4c78a8",
        label="Raw",
        edgecolor="white",
    )
    ax_hist.hist(
        refined_errors,
        bins=bins,
        alpha=0.6,
        color="#f58518",
        label="Refined",
        edgecolor="white",
    )
    ax_hist.set_xlabel(r"$|\Delta \theta|$ [arcsec]")
    ax_hist.set_ylabel("Number of matched images")
    ax_hist.set_xscale("log")
    ax_hist.set_title("Position error distribution")
    ax_hist.legend(frameon=False)
    ax_hist.grid(alpha=0.25)

    raw_plot = np.clip(raw_errors, floor, None)
    refined_plot = np.clip(refined_errors, floor, None)
    ax_scatter.scatter(raw_plot, refined_plot, s=18, alpha=0.55, color="#54a24b")
    diag_max = float(max(np.max(raw_plot), np.max(refined_plot))) if raw_plot.size else 1.0
    ax_scatter.plot([floor, diag_max], [floor, diag_max], "k--", lw=1.2, label="y = x")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel(r"Raw $|\Delta \theta|$ [arcsec]")
    ax_scatter.set_ylabel(r"Refined $|\Delta \theta|$ [arcsec]")
    ax_scatter.set_title("Refinement gain per matched image")
    ax_scatter.legend(frameon=False, loc="upper right")
    ax_scatter.grid(alpha=0.25, which="both")

    fig.suptitle("Analytic SIS image-finder validation", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, wspace=0.28)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_geometry(
    output_path: Path,
    df: deflector,
    source_positions: np.ndarray,
    analytic_positions: np.ndarray,
    numerical_positions: np.ndarray,
) -> None:
    """Plot the lens-plane critical curves, caustics, and image positions."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=False)
    ax_lens, ax_source = axes

    extent = (
        float(df.thetax[0]),
        float(df.thetax[-1]),
        float(df.thetay[0]),
        float(df.thetay[-1]),
    )

    kappa = np.asarray(df.ka, dtype=float)
    vmin = float(np.nanpercentile(kappa, 5.0))
    vmax = float(np.nanpercentile(kappa, 99.0))
    ax_lens.imshow(
        kappa,
        origin="lower",
        extent=extent,
        cmap="Greys_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    clt = np.atleast_1d(df.tancl(size_principale=0.0))
    clr = np.atleast_1d(df.radcl(size_principale=0.0))
    caut = np.atleast_1d(df.getCaustics(clt, buf_size=0.0))
    rad_caut = np.atleast_1d(df.getCaustics(clr, buf_size=0.0)) if clr.size else np.array([])

    for cl in clt:
        try:
            x, y = df.getCritPoints(cl)
            ax_lens.plot(x, y, color="#d62728", lw=1.0, alpha=0.7)
        except Exception:
            continue

    for cl in clr:
        try:
            x, y = df.getCritPoints(cl)
            ax_lens.plot(x, y, color="#1f77b4", lw=0.9, ls="--", alpha=0.45)
        except Exception:
            continue

    for cau in caut:
        try:
            x, y = df.getCausticPoints(cau)
            ax_source.plot(x, y, color="#d62728", lw=1.0, alpha=0.75)
        except Exception:
            continue

    for cau in rad_caut:
        try:
            x, y = df.getCausticPoints(cau)
            ax_source.plot(x, y, color="#1f77b4", lw=0.9, ls="--", alpha=0.45)
        except Exception:
            continue

    if source_positions.size:
        ax_source.scatter(
            source_positions[:, 0],
            source_positions[:, 1],
            s=16,
            color="#4c78a8",
            alpha=0.55,
            label="sampled sources",
        )
    ax_source.scatter([0.0], [0.0], marker="*", s=90, color="#d62728", label="caustic center")

    if analytic_positions.size and numerical_positions.size:
        for ana, num in zip(analytic_positions, numerical_positions):
            ax_lens.plot([ana[0], num[0]], [ana[1], num[1]], color="0.6", lw=0.5, alpha=0.25)
        ax_lens.scatter(
            analytic_positions[:, 0],
            analytic_positions[:, 1],
            s=18,
            marker="x",
            color="#1f77b4",
            alpha=0.8,
            label="analytic images",
        )
        ax_lens.scatter(
            numerical_positions[:, 0],
            numerical_positions[:, 1],
            s=22,
            facecolors="none",
            edgecolors="#ff7f0e",
            alpha=0.9,
            label="numerical images",
        )

    ax_lens.set_aspect("equal", adjustable="box")
    ax_source.set_aspect("equal", adjustable="box")
    ax_lens.set_xlabel(r"$\theta_1$ [arcsec]")
    ax_lens.set_ylabel(r"$\theta_2$ [arcsec]")
    ax_source.set_xlabel(r"$\beta_1$ [arcsec]")
    ax_source.set_ylabel(r"$\beta_2$ [arcsec]")
    ax_lens.set_title("Lens plane: convergence, critical curves, and images")
    ax_source.set_title("Source plane: caustics and sampled sources")
    ax_lens.legend(frameon=False, loc="upper right", fontsize=9, labelcolor="red")
    ax_source.legend(frameon=False, loc="upper right", fontsize=9, labelcolor="red")
    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle("Analytic SIS geometry and image matching", fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.90, wspace=0.18)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    fig_path, csv_path = resolve_output_paths(args.output)

    npix = int(args.npix)
    if npix % 2 == 0:
        npix += 1
        print(f"Adjusted npix to odd value {npix} so the grid is centered.")

    df, theta = build_analytic_sis_deflector(
        theta_e=float(args.theta_e),
        zl=float(args.zl),
        zs=float(args.zs),
        npix=npix,
        fov_factor=float(args.fov_factor),
    )
    fov = float(theta[-1] - theta[0])

    rng = np.random.default_rng(args.seed)
    accepted_rows: list[dict[str, float]] = []
    accepted_sources: list[dict[str, float]] = []
    raw_errors: list[float] = []
    refined_errors: list[float] = []
    raw_extra_images: list[int] = []
    refined_extra_images: list[int] = []

    counts = {
        "attempts": 0,
        "accepted_sources": 0,
        "rejected_missing_parity": 0,
        "rejected_parity": 0,
        "rejected_matching": 0,
    }

    while counts["accepted_sources"] < int(args.n_sources):
        counts["attempts"] += 1
        if counts["attempts"] > max(2000, 50 * int(args.n_sources)):
            raise RuntimeError(
                "Could not generate enough valid sources. Consider widening the source-radius range."
            )

        beta1, beta2, beta_radius = sample_source_position(
            rng,
            theta_e=float(args.theta_e),
            rmin_fraction=float(args.source_rmin_fraction),
            rmax_fraction=float(args.source_rmax_fraction),
        )
        beta_angle = float(np.arctan2(beta2, beta1))

        analytic_images = analytic_sis_images(beta1, beta2, float(args.theta_e))
        if len(analytic_images) != 2:
            continue
        for ana in analytic_images:
            ana["signature"] = jacobian_signature(df, ana["x"], ana["y"])

        ps_raw = compute_numerical_images(
            df=df,
            beta1=beta1,
            beta2=beta2,
            npix=npix,
            fov=fov,
            zs=float(args.zs),
            refine=False,
        )
        ps_refined = compute_numerical_images(
            df=df,
            beta1=beta1,
            beta2=beta2,
            npix=npix,
            fov=fov,
            zs=float(args.zs),
            refine=True,
        )

        raw_images = [
            {
                "x": float(x),
                "y": float(y),
                "mu": float(mu),
                "parity": int(np.sign(mu)),
            }
            for x, y, mu in zip(np.asarray(ps_raw.xi1), np.asarray(ps_raw.xi2), np.asarray(ps_raw.mui))
        ]
        refined_images = [
            {
                "x": float(x),
                "y": float(y),
                "mu": float(mu),
                "parity": int(np.sign(mu)),
            }
            for x, y, mu in zip(np.asarray(ps_refined.xi1), np.asarray(ps_refined.xi2), np.asarray(ps_refined.mui))
        ]
        for num in raw_images:
            num["signature"] = jacobian_signature(df, num["x"], num["y"])
        for num in refined_images:
            num["signature"] = jacobian_signature(df, num["x"], num["y"])

        raw_matched = match_images_by_signature(analytic_images, raw_images)
        refined_matched = match_images_by_signature(analytic_images, refined_images)

        if len(raw_matched) != 2 or len(refined_matched) != 2:
            counts["rejected_missing_parity"] += 1
            continue

        if any(tuple(int(v) for v in ana["signature"]) != tuple(int(v) for v in num["signature"]) for ana, num in raw_matched):
            counts["rejected_parity"] += 1
            continue
        if any(tuple(int(v) for v in ana["signature"]) != tuple(int(v) for v in num["signature"]) for ana, num in refined_matched):
            counts["rejected_parity"] += 1
            continue

        source_index = counts["accepted_sources"]
        counts["accepted_sources"] += 1
        raw_extra_images.append(max(0, len(raw_images) - 2))
        refined_extra_images.append(max(0, len(refined_images) - 2))
        accepted_sources.append(
            {
                "source_index": source_index,
                "beta1_arcsec": float(beta1),
                "beta2_arcsec": float(beta2),
            }
        )

        for ana, num_raw in raw_matched:
            num_ref = next(
                num
                for ana2, num in refined_matched
                if tuple(int(v) for v in ana2["signature"]) == tuple(int(v) for v in ana["signature"])
            )

            raw_dx = float(num_raw["x"] - ana["x"])
            raw_dy = float(num_raw["y"] - ana["y"])
            refined_dx = float(num_ref["x"] - ana["x"])
            refined_dy = float(num_ref["y"] - ana["y"])
            raw_dr = float(np.hypot(raw_dx, raw_dy))
            refined_dr = float(np.hypot(refined_dx, refined_dy))

            raw_errors.append(raw_dr)
            refined_errors.append(refined_dr)

            accepted_rows.append(
                {
                    "source_index": source_index,
                    "beta1_arcsec": float(beta1),
                    "beta2_arcsec": float(beta2),
                    "beta_radius_arcsec": float(beta_radius),
                    "beta_angle_rad": float(beta_angle),
                    "n_images_raw": int(len(raw_images)),
                    "n_images_refined": int(len(refined_images)),
                    "n_extra_images_raw": int(max(0, len(raw_images) - 2)),
                    "n_extra_images_refined": int(max(0, len(refined_images) - 2)),
                    "signature_1": int(ana["signature"][0]),
                    "signature_2": int(ana["signature"][1]),
                    "analytic_x_arcsec": float(ana["x"]),
                    "analytic_y_arcsec": float(ana["y"]),
                    "analytic_mu": float(ana["mu"]),
                    "raw_x_arcsec": float(num_raw["x"]),
                    "raw_y_arcsec": float(num_raw["y"]),
                    "raw_mu": float(num_raw["mu"]),
                    "refined_x_arcsec": float(num_ref["x"]),
                    "refined_y_arcsec": float(num_ref["y"]),
                    "refined_mu": float(num_ref["mu"]),
                    "raw_dx_arcsec": raw_dx,
                    "raw_dy_arcsec": raw_dy,
                    "raw_dr_arcsec": raw_dr,
                    "refined_dx_arcsec": refined_dx,
                    "refined_dy_arcsec": refined_dy,
                    "refined_dr_arcsec": refined_dr,
                }
            )

    raw_errors_arr = np.asarray(raw_errors, dtype=float)
    refined_errors_arr = np.asarray(refined_errors, dtype=float)
    source_positions = np.array(
        [(src["beta1_arcsec"], src["beta2_arcsec"]) for src in accepted_sources], dtype=float
    )
    analytic_positions = np.array(
        [(row["analytic_x_arcsec"], row["analytic_y_arcsec"]) for row in accepted_rows], dtype=float
    )
    numerical_positions = np.array(
        [(row["refined_x_arcsec"], row["refined_y_arcsec"]) for row in accepted_rows], dtype=float
    )

    write_catalog(accepted_rows, csv_path)
    plot_summary(fig_path, raw_errors_arr, refined_errors_arr)
    geometry_path = fig_path.with_name(f"{fig_path.stem}_geometry{fig_path.suffix}")
    plot_geometry(geometry_path, df, source_positions, analytic_positions, numerical_positions)

    source_stats_raw = summarize_errors(raw_errors_arr)
    source_stats_refined = summarize_errors(refined_errors_arr)
    mean_raw_extra = float(np.mean(raw_extra_images)) if raw_extra_images else np.nan
    mean_refined_extra = float(np.mean(refined_extra_images)) if refined_extra_images else np.nan

    print(f"Accepted sources: {counts['accepted_sources']}")
    print(f"Total matched images: {len(accepted_rows)}")
    print(f"Rejected sources due to missing parity match: {counts['rejected_missing_parity']}")
    print(f"Rejected sources due to parity mismatch: {counts['rejected_parity']}")
    print(f"Rejected sources due to matching failure: {counts['rejected_matching']}")
    print(f"Mean extra numerical images per accepted source (raw): {mean_raw_extra:.3f}")
    print(f"Mean extra numerical images per accepted source (refined): {mean_refined_extra:.3f}")
    print(
        "Raw positional error [arcsec]: "
        f"mean={source_stats_raw['mean']:.6g}, median={source_stats_raw['median']:.6g}, "
        f"rms={source_stats_raw['rms']:.6g}, max={source_stats_raw['max']:.6g}"
    )
    print(
        "Refined positional error [arcsec]: "
        f"mean={source_stats_refined['mean']:.6g}, median={source_stats_refined['median']:.6g}, "
        f"rms={source_stats_refined['rms']:.6g}, max={source_stats_refined['max']:.6g}"
    )
    if source_stats_raw["rms"] > 0.0 and np.isfinite(source_stats_raw["rms"]) and np.isfinite(source_stats_refined["rms"]):
        print(f"RMS improvement factor: {source_stats_raw['rms'] / source_stats_refined['rms']:.3f}")

    print(f"Saved figure to {fig_path}")
    print(f"Saved geometry figure to {geometry_path}")
    print(f"Saved catalog to {csv_path}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
