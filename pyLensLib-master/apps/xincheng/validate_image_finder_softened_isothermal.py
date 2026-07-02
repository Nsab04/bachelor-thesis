#!/usr/bin/env python3
"""Validate the point-source image finder with a softened isothermal sphere.

The softened isothermal potential used here is

    psi(theta) = b * sqrt(theta^2 + core^2)

so the circular deflection is

    alpha(theta) = b * theta / sqrt(theta^2 + core^2).

For each source, the reference image positions are found from the circular
image diagram by solving beta = theta - alpha(theta) along the source-lens
axis.  Those positions are then compared with the raw and refined ``pointsrc``
solutions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq

from pyLensLib.deflector import deflector
from pyLensLib.pointsrc import pointsrc


DEFAULT_OUTPUT = Path("validate_image_finder_softened_isothermal.png")


def resolve_output_paths(output_arg: str | None) -> tuple[Path, Path]:
    output = DEFAULT_OUTPUT if output_arg is None else Path(output_arg)
    if output.suffix:
        fig_path = output
        out_dir = output.parent
        stem = output.stem
    else:
        out_dir = output
        stem = "validate_image_finder_softened_isothermal"
        fig_path = out_dir / f"{stem}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    return fig_path, out_dir / f"{stem}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate image finding against image-diagram roots for a softened isothermal sphere."
    )
    parser.add_argument("--n-sources", type=int, default=100, help="Number of accepted sources. Default: 100")
    parser.add_argument("--zl", type=float, default=0.5, help="Lens redshift. Default: 0.5")
    parser.add_argument("--zs", type=float, default=2.0, help="Source redshift. Default: 2.0")
    parser.add_argument("--einstein-radius", type=float, default=1.5, help="Strength b in arcsec. Default: 1.5")
    parser.add_argument("--core-radius", type=float, default=0.15, help="Core radius in arcsec. Default: 0.15")
    parser.add_argument("--npix", type=int, default=1001, help="Grid size. Default: 1001")
    parser.add_argument("--fov-factor", type=float, default=8.0, help="Field of view in units of b. Default: 8")
    parser.add_argument("--source-rmin-fraction", type=float, default=0.05, help="Minimum source radius as fraction of caustic. Default: 0.05")
    parser.add_argument("--source-rmax-fraction", type=float, default=0.85, help="Maximum source radius as fraction of caustic. Default: 0.85")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed. Default: 12345")
    parser.add_argument("--root-samples", type=int, default=4000, help="Signed-theta samples for root bracketing. Default: 4000")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output figure path or directory.")
    parser.add_argument("--show", dest="show", action="store_true", help="Display figures interactively.")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Only save figures.")
    parser.set_defaults(show=False)
    return parser.parse_args()


def alpha_radius(radius: np.ndarray | float, strength: float, core: float) -> np.ndarray:
    r = np.asarray(radius, dtype=float)
    scalar = r.ndim == 0
    out = strength * r / np.sqrt(r**2 + core**2)
    return out.item() if scalar else out


def build_deflector(
    strength: float,
    core: float,
    zl: float,
    zs: float,
    npix: int,
    fov_factor: float,
) -> tuple[deflector, np.ndarray]:
    co = FlatLambdaCDM(H0=70.0, Om0=0.3)
    fov = fov_factor * strength
    theta = np.linspace(-0.5 * fov, 0.5 * fov, npix)
    theta1, theta2 = np.meshgrid(theta, theta)
    potential = strength * np.sqrt(theta1**2 + theta2**2 + core**2)
    df = deflector(co, pot=potential, usePotential=True, zl=zl, zs=zs)
    df.setGrid(theta=theta, compute_potential=False)
    return df, theta


def signed_lens_equation(theta: float, beta: float, strength: float, core: float) -> float:
    return theta - strength * theta / np.sqrt(theta**2 + core**2) - beta


def image_diagram_roots(
    beta: float,
    strength: float,
    core: float,
    theta_max: float,
    n_samples: int,
) -> list[float]:
    grid = np.linspace(-theta_max, theta_max, int(n_samples))
    values = signed_lens_equation(grid, beta, strength, core)
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0:
            root = float(left)
        elif f_left * f_right < 0.0:
            root = float(
                brentq(
                    signed_lens_equation,
                    left,
                    right,
                    args=(beta, strength, core),
                    xtol=1e-12,
                    rtol=1e-12,
                    maxiter=100,
                )
            )
        else:
            continue
        if all(abs(root - existing) > 1e-6 for existing in roots):
            roots.append(root)
    return sorted(roots)


def estimate_caustic_radius(strength: float, core: float, theta_max: float) -> float:
    if strength <= core:
        raise RuntimeError("The softened isothermal sphere needs b > core to produce multiple images.")
    theta_crit = np.sqrt((strength * core**2) ** (2.0 / 3.0) - core**2)
    beta_crit = abs(signed_lens_equation(theta_crit, 0.0, strength, core))
    if theta_crit >= theta_max or beta_crit <= 0.0:
        raise RuntimeError("Field of view is too small for the softened-isothermal caustic.")
    return float(beta_crit)


def sample_source_position(
    rng: np.random.Generator,
    beta_caustic: float,
    rmin_fraction: float,
    rmax_fraction: float,
) -> tuple[float, float, float]:
    rmin = max(0.0, rmin_fraction * beta_caustic)
    rmax = min(0.999 * beta_caustic, rmax_fraction * beta_caustic)
    if rmax <= rmin:
        raise ValueError("source-rmax-fraction must be larger than source-rmin-fraction.")
    radius = np.sqrt(rng.uniform(rmin**2, rmax**2))
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return float(radius * np.cos(angle)), float(radius * np.sin(angle)), float(radius)


def sample_field(df: deflector, field: np.ndarray, x_arcsec: float, y_arcsec: float) -> float:
    xpix = float(x_arcsec) / df.pixel_scale + (df.nray1 - 1) / 2.0
    ypix = float(y_arcsec) / df.pixel_scale + (df.nray2 - 1) / 2.0
    return float(map_coordinates(field, [[ypix], [xpix]], order=1, mode="nearest")[0])


def jacobian_signature(df: deflector, x_arcsec: float, y_arcsec: float) -> tuple[int, int]:
    kappa = sample_field(df, df.ka, x_arcsec, y_arcsec)
    g1 = sample_field(df, df.g1, x_arcsec, y_arcsec)
    g2 = sample_field(df, df.g2, x_arcsec, y_arcsec)
    jac = np.array([[1.0 - kappa - g1, -g2], [-g2, 1.0 - kappa + g1]], dtype=float)
    eigvals = np.linalg.eigvalsh(jac)
    return tuple(int(np.sign(val)) for val in eigvals)


def roots_to_positions(
    roots: list[float],
    beta1: float,
    beta2: float,
    df: deflector,
) -> list[dict[str, float]]:
    beta = float(np.hypot(beta1, beta2))
    ux = beta1 / beta
    uy = beta2 / beta
    images: list[dict[str, float]] = []
    for root in roots:
        x = root * ux
        y = root * uy
        images.append({"x": float(x), "y": float(y), "theta": float(root), "signature": jacobian_signature(df, x, y)})
    return images


def compute_numerical_images(
    df: deflector,
    beta1: float,
    beta2: float,
    npix: int,
    fov: float,
    zs: float,
    refine: bool,
) -> pointsrc:
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


def match_by_signature_and_position(
    reference_images: list[dict[str, float]],
    numeric_images: list[dict[str, float]],
) -> list[tuple[dict[str, float], dict[str, float]]]:
    matched: list[tuple[dict[str, float], dict[str, float]]] = []
    used: set[int] = set()
    for ref in reference_images:
        candidates = [
            (idx, img)
            for idx, img in enumerate(numeric_images)
            if idx not in used and tuple(img["signature"]) == tuple(ref["signature"])
        ]
        if not candidates:
            return []
        idx, img = min(
            candidates,
            key=lambda item: (item[1]["x"] - ref["x"]) ** 2 + (item[1]["y"] - ref["y"]) ** 2,
        )
        used.add(idx)
        matched.append((ref, img))
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
        "n_reference_images",
        "n_images_raw",
        "n_images_refined",
        "signature_1",
        "signature_2",
        "reference_theta_arcsec",
        "reference_x_arcsec",
        "reference_y_arcsec",
        "raw_x_arcsec",
        "raw_y_arcsec",
        "refined_x_arcsec",
        "refined_y_arcsec",
        "raw_dx_arcsec",
        "raw_dy_arcsec",
        "raw_dr_arcsec",
        "refined_dx_arcsec",
        "refined_dy_arcsec",
        "refined_dr_arcsec",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output_path: Path, raw_errors: np.ndarray, refined_errors: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=False)
    ax_hist, ax_scatter = axes
    all_errors = np.concatenate([raw_errors, refined_errors]) if raw_errors.size or refined_errors.size else np.array([])
    positive = all_errors[all_errors > 0.0]
    floor = max(1e-10, 0.5 * float(np.min(positive))) if positive.size else 1e-10
    top = max(float(np.max(all_errors)) * 1.05 if all_errors.size else 1.0, floor * 10.0)
    bins = np.logspace(np.log10(floor), np.log10(top), 36)
    ax_hist.hist(raw_errors, bins=bins, alpha=0.6, color="#4c78a8", edgecolor="white", label="Raw")
    ax_hist.hist(refined_errors, bins=bins, alpha=0.6, color="#f58518", edgecolor="white", label="Refined")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel(r"$|\Delta \theta|$ [arcsec]")
    ax_hist.set_ylabel("Number of matched images")
    ax_hist.set_title("Softened-isothermal image-position offsets")
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
    ax_scatter.set_title("Refinement gain")
    ax_scatter.legend(frameon=False)
    ax_scatter.grid(alpha=0.25, which="both")
    fig.suptitle("Softened isothermal sphere image-finder validation", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, wspace=0.28)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_image_diagram(
    output_path: Path,
    strength: float,
    core: float,
    beta_caustic: float,
    theta_max: float,
) -> None:
    theta = np.linspace(-theta_max, theta_max, 4000)
    alpha_signed = strength * theta / np.sqrt(theta**2 + core**2)
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot(theta, alpha_signed, color="tab:blue", lw=1.8, label=r"$\alpha(\theta)$")
    for beta, color in [
        (0.25 * beta_caustic, "tab:orange"),
        (0.60 * beta_caustic, "tab:green"),
        (0.95 * beta_caustic, "tab:red"),
    ]:
        ax.plot(theta, theta - beta, color=color, lw=1.0, ls="--", label=rf"$\theta-\beta$, $\beta={beta:.2f}$")
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.axvline(0.0, color="0.75", lw=0.8)
    ax.set_xlabel(r"signed $\theta$ [arcsec]")
    ax.set_ylabel("image diagram ordinate [arcsec]")
    ax.set_title("Softened isothermal sphere image diagram")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    fig_path, csv_path = resolve_output_paths(args.output)
    npix = int(args.npix)
    if npix % 2 == 0:
        npix += 1
        print(f"Adjusted npix to odd value {npix} so the grid is centered.")

    strength = float(args.einstein_radius)
    core = float(args.core_radius)
    df, theta = build_deflector(strength, core, float(args.zl), float(args.zs), npix, float(args.fov_factor))
    fov = float(theta[-1] - theta[0])
    theta_max = 0.5 * fov
    beta_caustic = estimate_caustic_radius(strength, core, theta_max)
    print(f"Estimated softened-isothermal radial caustic radius: {beta_caustic:.6f} arcsec")

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, float]] = []
    raw_errors: list[float] = []
    refined_errors: list[float] = []
    counts = {"attempts": 0, "accepted_sources": 0, "rejected_matching": 0, "rejected_image_count": 0}

    while counts["accepted_sources"] < int(args.n_sources):
        counts["attempts"] += 1
        if counts["attempts"] > max(2000, 80 * int(args.n_sources)):
            raise RuntimeError("Could not generate enough valid softened-isothermal multiple-image sources.")
        beta1, beta2, beta_radius = sample_source_position(
            rng,
            beta_caustic,
            float(args.source_rmin_fraction),
            float(args.source_rmax_fraction),
        )
        beta = float(np.hypot(beta1, beta2))
        roots = image_diagram_roots(beta, strength, core, theta_max, int(args.root_samples))
        if len(roots) < 2:
            counts["rejected_image_count"] += 1
            continue
        reference_images = roots_to_positions(roots, beta1, beta2, df)

        ps_raw = compute_numerical_images(df, beta1, beta2, npix, fov, float(args.zs), refine=False)
        ps_refined = compute_numerical_images(df, beta1, beta2, npix, fov, float(args.zs), refine=True)
        raw_images = [
            {"x": float(x), "y": float(y), "signature": jacobian_signature(df, float(x), float(y))}
            for x, y in zip(np.asarray(ps_raw.xi1), np.asarray(ps_raw.xi2))
        ]
        refined_images = [
            {"x": float(x), "y": float(y), "signature": jacobian_signature(df, float(x), float(y))}
            for x, y in zip(np.asarray(ps_refined.xi1), np.asarray(ps_refined.xi2))
        ]
        raw_matched = match_by_signature_and_position(reference_images, raw_images)
        refined_matched = match_by_signature_and_position(reference_images, refined_images)
        if len(raw_matched) != len(reference_images) or len(refined_matched) != len(reference_images):
            counts["rejected_matching"] += 1
            continue

        source_index = counts["accepted_sources"]
        counts["accepted_sources"] += 1
        refined_by_sig = {tuple(ref["signature"]): num for ref, num in refined_matched}
        for ref, raw in raw_matched:
            refined = refined_by_sig[tuple(ref["signature"])]
            raw_dx = float(raw["x"] - ref["x"])
            raw_dy = float(raw["y"] - ref["y"])
            refined_dx = float(refined["x"] - ref["x"])
            refined_dy = float(refined["y"] - ref["y"])
            raw_dr = float(np.hypot(raw_dx, raw_dy))
            refined_dr = float(np.hypot(refined_dx, refined_dy))
            raw_errors.append(raw_dr)
            refined_errors.append(refined_dr)
            rows.append(
                {
                    "source_index": source_index,
                    "beta1_arcsec": float(beta1),
                    "beta2_arcsec": float(beta2),
                    "beta_radius_arcsec": float(beta_radius),
                    "n_reference_images": int(len(reference_images)),
                    "n_images_raw": int(len(raw_images)),
                    "n_images_refined": int(len(refined_images)),
                    "signature_1": int(ref["signature"][0]),
                    "signature_2": int(ref["signature"][1]),
                    "reference_theta_arcsec": float(ref["theta"]),
                    "reference_x_arcsec": float(ref["x"]),
                    "reference_y_arcsec": float(ref["y"]),
                    "raw_x_arcsec": float(raw["x"]),
                    "raw_y_arcsec": float(raw["y"]),
                    "refined_x_arcsec": float(refined["x"]),
                    "refined_y_arcsec": float(refined["y"]),
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
    write_catalog(rows, csv_path)
    plot_summary(fig_path, raw_errors_arr, refined_errors_arr)
    diagram_path = fig_path.with_name(f"{fig_path.stem}_image_diagram{fig_path.suffix}")
    plot_image_diagram(diagram_path, strength, core, beta_caustic, theta_max)

    raw_stats = summarize_errors(raw_errors_arr)
    refined_stats = summarize_errors(refined_errors_arr)
    print(f"Accepted sources: {counts['accepted_sources']}")
    print(f"Total matched images: {len(rows)}")
    print(f"Rejected sources due to reference image count: {counts['rejected_image_count']}")
    print(f"Rejected sources due to matching failure: {counts['rejected_matching']}")
    print(
        "Raw positional error [arcsec]: "
        f"mean={raw_stats['mean']:.6g}, median={raw_stats['median']:.6g}, "
        f"rms={raw_stats['rms']:.6g}, max={raw_stats['max']:.6g}"
    )
    print(
        "Refined positional error [arcsec]: "
        f"mean={refined_stats['mean']:.6g}, median={refined_stats['median']:.6g}, "
        f"rms={refined_stats['rms']:.6g}, max={refined_stats['max']:.6g}"
    )
    if raw_stats["rms"] > 0.0 and np.isfinite(raw_stats["rms"]) and np.isfinite(refined_stats["rms"]):
        print(f"RMS improvement factor: {raw_stats['rms'] / refined_stats['rms']:.3f}")
    print(f"Saved figure to {fig_path}")
    print(f"Saved image diagram to {diagram_path}")
    print(f"Saved catalog to {csv_path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
