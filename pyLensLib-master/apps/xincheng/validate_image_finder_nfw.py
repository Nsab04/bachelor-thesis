#!/usr/bin/env python3
"""Validate the point-source image finder against a circular NFW lens.

For a circular lens, the image positions of a source lie on the source-lens
axis.  The reference solution used here is therefore the image diagram: find
all signed intersections of

    beta = theta - alpha(theta)

where alpha(theta) is the circular NFW deflection profile.  The resulting
1D roots are converted back into 2D image positions and compared with the
numerical positions returned by ``pointsrc``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import brentq

from pyLensLib.deflector import deflector
from pyLensLib.pointsrc import pointsrc


DEFAULT_OUTPUT = Path("validate_image_finder_nfw.png")


def resolve_output_paths(output_arg: str | None) -> tuple[Path, Path]:
    output = DEFAULT_OUTPUT if output_arg is None else Path(output_arg)
    if output.suffix:
        fig_path = output
        out_dir = output.parent
        stem = output.stem
    else:
        out_dir = output
        stem = "validate_image_finder_nfw"
        fig_path = out_dir / f"{stem}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    return fig_path, out_dir / f"{stem}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the image finder against image-diagram roots for a circular NFW lens."
    )
    parser.add_argument("--n-sources", type=int, default=100, help="Number of accepted sources. Default: 100")
    parser.add_argument("--zl", type=float, default=0.5, help="Lens redshift. Default: 0.5")
    parser.add_argument("--zs", type=float, default=2.0, help="Source redshift. Default: 2.0")
    parser.add_argument("--theta-s", type=float, default=3.0, help="NFW angular scale radius in arcsec. Default: 3.0")
    parser.add_argument("--kappa-s", type=float, default=1.2, help="NFW scale convergence. Default: 1.2")
    parser.add_argument("--npix", type=int, default=1001, help="Grid size. Default: 1001")
    parser.add_argument("--fov-factor", type=float, default=12.0, help="Field of view in units of theta_s. Default: 12")
    parser.add_argument("--source-rmin-fraction", type=float, default=0.05, help="Minimum source radius as fraction of radial caustic. Default: 0.05")
    parser.add_argument("--source-rmax-fraction", type=float, default=0.80, help="Maximum source radius as fraction of radial caustic. Default: 0.80")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed. Default: 12345")
    parser.add_argument("--root-samples", type=int, default=4000, help="Signed-theta samples for root bracketing. Default: 4000")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output figure path or directory.")
    parser.add_argument("--show", dest="show", action="store_true", help="Display figures interactively.")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Only save figures.")
    parser.set_defaults(show=False)
    return parser.parse_args()


def nfw_g(x: np.ndarray | float) -> np.ndarray:
    """Dimensionless circular NFW deflection helper."""
    x_arr = np.asarray(x, dtype=float)
    scalar = x_arr.ndim == 0
    x_safe = np.maximum(x_arr, 1e-12)
    out = np.empty_like(x_safe, dtype=float)

    small = x_safe < 1e-5
    below = (x_safe < 1.0) & ~small
    above = x_safe > 1.0
    equal = np.isclose(x_safe, 1.0, rtol=0.0, atol=1e-8)

    out[small] = 0.5 * x_safe[small] ** 2 * (np.log(2.0 / x_safe[small]) + 0.5)
    xb = x_safe[below]
    out[below] = np.log(xb / 2.0) + np.arctanh(np.sqrt(1.0 - xb**2)) / np.sqrt(1.0 - xb**2)
    xa = x_safe[above & ~equal]
    out[above & ~equal] = np.log(xa / 2.0) + np.arctan(np.sqrt(xa**2 - 1.0)) / np.sqrt(xa**2 - 1.0)
    out[equal] = np.log(0.5) + 1.0
    return out.item() if scalar else out


def nfw_alpha_radius(r: np.ndarray | float, theta_s: float, kappa_s: float) -> np.ndarray:
    r_arr = np.asarray(r, dtype=float)
    scalar = r_arr.ndim == 0
    x = np.maximum(r_arr / theta_s, 1e-12)
    alpha = 4.0 * kappa_s * theta_s * nfw_g(x) / x
    alpha = np.where(r_arr <= 0.0, 0.0, alpha)
    return alpha.item() if scalar else alpha


def build_nfw_deflector(
    theta_s: float,
    kappa_s: float,
    zl: float,
    zs: float,
    npix: int,
    fov_factor: float,
) -> tuple[deflector, np.ndarray]:
    co = FlatLambdaCDM(H0=70.0, Om0=0.3)
    fov = fov_factor * theta_s
    theta = np.linspace(-0.5 * fov, 0.5 * fov, npix)
    theta1, theta2 = np.meshgrid(theta, theta)
    radius = np.hypot(theta1, theta2)
    alpha = nfw_alpha_radius(radius, theta_s=theta_s, kappa_s=kappa_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        ax = np.divide(alpha * theta1, radius, out=np.zeros_like(theta1), where=radius > 0.0)
        ay = np.divide(alpha * theta2, radius, out=np.zeros_like(theta2), where=radius > 0.0)
    df = deflector(co, angx=ax, angy=ay, zl=zl, zs=zs)
    df.setGrid(theta=theta, compute_potential=True)
    return df, theta


def signed_lens_equation(theta: float, beta: float, theta_s: float, kappa_s: float) -> float:
    if theta == 0.0:
        return -beta
    return theta - np.sign(theta) * float(nfw_alpha_radius(abs(theta), theta_s, kappa_s)) - beta


def image_diagram_roots(
    beta: float,
    theta_s: float,
    kappa_s: float,
    theta_max: float,
    n_samples: int,
) -> list[float]:
    eps = max(theta_max * 1e-8, 1e-8)
    neg = -np.geomspace(theta_max, eps, n_samples // 2)
    pos = np.geomspace(eps, theta_max, n_samples // 2)
    grid = np.concatenate([neg, pos])
    values = np.array([signed_lens_equation(t, beta, theta_s, kappa_s) for t in grid], dtype=float)
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0:
            root = left
        elif f_left * f_right < 0.0:
            root = brentq(
                signed_lens_equation,
                left,
                right,
                args=(beta, theta_s, kappa_s),
                xtol=1e-12,
                rtol=1e-12,
                maxiter=100,
            )
        else:
            continue
        if all(abs(root - existing) > 1e-6 for existing in roots):
            roots.append(float(root))
    return sorted(roots)


def estimate_radial_caustic(theta_s: float, kappa_s: float, theta_max: float) -> float:
    radii = np.geomspace(theta_max * 1e-5, theta_max, 20000)
    beta_curve = radii - nfw_alpha_radius(radii, theta_s, kappa_s)
    negative = beta_curve[beta_curve < 0.0]
    if negative.size == 0:
        raise RuntimeError("This NFW lens is not strong enough to produce multiple images.")
    return float(abs(np.min(negative)))


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


def image_roots_to_positions(
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
        images.append({"x": float(x), "y": float(y), "signature": jacobian_signature(df, x, y)})
    return images


def sample_field(df: deflector, field: np.ndarray, x_arcsec: float, y_arcsec: float) -> float:
    from scipy.ndimage import map_coordinates

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
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    ax_hist.set_title("NFW image-position offsets")
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

    fig.suptitle("Circular NFW image-finder validation", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, wspace=0.28)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_image_diagram(
    output_path: Path,
    theta_s: float,
    kappa_s: float,
    beta_caustic: float,
    theta_max: float,
) -> None:
    theta = np.linspace(-theta_max, theta_max, 4000)
    alpha_signed = np.sign(theta) * nfw_alpha_radius(np.abs(theta), theta_s, kappa_s)
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
    ax.set_title("Circular NFW image diagram")
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

    theta_s = float(args.theta_s)
    kappa_s = float(args.kappa_s)
    df, theta = build_nfw_deflector(theta_s, kappa_s, float(args.zl), float(args.zs), npix, float(args.fov_factor))
    fov = float(theta[-1] - theta[0])
    theta_max = 0.5 * fov
    beta_caustic = estimate_radial_caustic(theta_s, kappa_s, theta_max)
    print(f"Estimated NFW radial caustic radius: {beta_caustic:.6f} arcsec")

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, float]] = []
    raw_errors: list[float] = []
    refined_errors: list[float] = []
    counts = {"attempts": 0, "accepted_sources": 0, "rejected_matching": 0, "rejected_image_count": 0}

    while counts["accepted_sources"] < int(args.n_sources):
        counts["attempts"] += 1
        if counts["attempts"] > max(2000, 80 * int(args.n_sources)):
            raise RuntimeError("Could not generate enough valid NFW multiple-image sources.")

        beta1, beta2, beta_radius = sample_source_position(
            rng,
            beta_caustic,
            float(args.source_rmin_fraction),
            float(args.source_rmax_fraction),
        )
        beta = float(np.hypot(beta1, beta2))
        roots = image_diagram_roots(beta, theta_s, kappa_s, theta_max, int(args.root_samples))
        if len(roots) < 2:
            counts["rejected_image_count"] += 1
            continue
        reference_images = image_roots_to_positions(roots, beta1, beta2, df)

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
                    "reference_theta_arcsec": float(np.sign(ref["x"] * beta1 + ref["y"] * beta2) * np.hypot(ref["x"], ref["y"])),
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
    plot_image_diagram(diagram_path, theta_s, kappa_s, beta_caustic, theta_max)

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
