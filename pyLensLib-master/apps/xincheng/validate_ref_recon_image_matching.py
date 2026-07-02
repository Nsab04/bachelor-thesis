#!/usr/bin/env python3
"""
Make a focused REF/RECON image-matching diagnostic for a few multiply imaged
sources.

The plot shows the lens plane for each accepted source with REF and RECON
critical lines, matched image positions, image labels ordered by REF arrival
time, and the REF-to-RECON positional offsets for each matched image.
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.xincheng.plot_source_image_displacements import (
    DEFAULT_INPUT_DIR,
    DEFAULT_ZS_COLUMN,
    DEFAULT_ZS_SAMPLES,
    build_deflector_for_case,
    display_critical_lines,
    load_case,
    load_empirical_redshifts,
    match_source_images,
    relative_time_delays_for_images,
    resolve_output_path,
    sample_empirical_redshifts,
    sample_source_inside_ref_caustics,
)


DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "ref_recon_image_matching_validation.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate REF/RECON image matching with labeled multi-image examples."
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
    parser.add_argument("--zl", type=float, required=True, help="Lens redshift.")
    parser.add_argument("--pixel-scale-arcsec", type=float, required=True, help="Deflection map pixel scale.")
    parser.add_argument("--zs-norm", type=float, default=1.6, help="Source redshift assumed by the maps. Default: 1.6")
    parser.add_argument("--h0", type=float, default=70.0, help="Hubble constant. Default: 70")
    parser.add_argument("--om0", type=float, default=0.3, help="Matter density. Default: 0.3")
    parser.add_argument("--n-sources", type=int, default=3, help="Number of accepted sources to plot. Default: 3")
    parser.add_argument("--seed", type=int, default=902, help="Random seed. Default: 902")
    parser.add_argument(
        "--zs-samples-file",
        type=Path,
        default=DEFAULT_ZS_SAMPLES,
        help=f"CSV file containing the empirical source-redshift sample. Default: {DEFAULT_ZS_SAMPLES}",
    )
    parser.add_argument(
        "--zs-column",
        type=str,
        default=DEFAULT_ZS_COLUMN,
        help=f"Redshift column in --zs-samples-file. Default: {DEFAULT_ZS_COLUMN}",
    )
    parser.add_argument(
        "--match-max-sep-arcsec",
        type=float,
        default=2.0,
        help="Maximum allowed separation for position+parity matching. Default: 2.0",
    )
    parser.add_argument(
        "--caustic-buffer-arcsec",
        type=float,
        default=0.0,
        help="Optional source-plane caustic buffer. Default: 0.0",
    )
    parser.add_argument(
        "--max-attempts-factor",
        type=int,
        default=80,
        help="Source-position attempts per sampled redshift. Default: 80",
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show", dest="show", action="store_true", help="Show figure interactively.")
    show_group.add_argument("--no-show", dest="show", action="store_false", help="Only save the figure.")
    parser.set_defaults(show=False)
    return parser.parse_args()


def image_label(index: int) -> str:
    letters = string.ascii_uppercase
    label = ""
    n = int(index)
    while True:
        n, rem = divmod(n, len(letters))
        label = letters[rem] + label
        if n == 0:
            break
        n -= 1
    return label


def collect_examples(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], object, object]:
    input_dir = args.input_dir.expanduser().resolve()
    co = FlatLambdaCDM(H0=float(args.h0), Om0=float(args.om0))
    ref_ax, ref_ay = load_case(input_dir, "ref")
    recon_ax, recon_ay = load_case(input_dir, "recon")
    if ref_ax.shape != recon_ax.shape:
        raise ValueError(f"REF and RECON shapes differ: {ref_ax.shape} vs {recon_ax.shape}")

    zs_samples = load_empirical_redshifts(args.zs_samples_file.expanduser().resolve(), args.zs_column)
    rng = np.random.default_rng(int(args.seed))
    requested = int(args.n_sources)
    sampled_zs = sample_empirical_redshifts(
        zs_samples,
        rng,
        max(requested * int(args.max_attempts_factor), requested),
        float(args.zl),
    )

    examples: list[dict[str, object]] = []
    skipped = 0
    first_ref = None
    first_recon = None

    for z_target in sampled_zs:
        if len(examples) >= requested:
            break
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
        if ref_ps is None or recon_ps is None or not sampled_from_shared:
            skipped += 1
            continue

        ref_x = np.asarray(ref_ps.xi1, dtype=float)
        ref_y = np.asarray(ref_ps.xi2, dtype=float)
        recon_x = np.asarray(recon_ps.xi1, dtype=float)
        recon_y = np.asarray(recon_ps.xi2, dtype=float)
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
            skipped += 1
            continue

        row_sel, col_sel = matched
        ref_x = ref_x[row_sel]
        ref_y = ref_y[row_sel]
        recon_x = recon_x[col_sel]
        recon_y = recon_y[col_sel]
        ref_td_raw, _ = relative_time_delays_for_images(df_ref, beta, ref_x, ref_y)
        recon_td_raw, _ = relative_time_delays_for_images(df_recon, beta, recon_x, recon_y)
        if ref_td_raw.size != ref_x.size or recon_td_raw.size != recon_x.size:
            skipped += 1
            continue
        if not np.all(np.isfinite(ref_td_raw)) or not np.all(np.isfinite(recon_td_raw)):
            skipped += 1
            continue

        order = np.argsort(ref_td_raw)
        ref_x = ref_x[order]
        ref_y = ref_y[order]
        recon_x = recon_x[order]
        recon_y = recon_y[order]
        ref_td_raw = ref_td_raw[order]
        recon_td_raw = recon_td_raw[order]
        labels = [image_label(i) for i in range(ref_x.size)]
        offsets = np.sqrt((recon_x - ref_x) ** 2 + (recon_y - ref_y) ** 2)

        examples.append(
            {
                "z": z_target,
                "beta": beta,
                "df_ref": df_ref,
                "df_recon": df_recon,
                "labels": labels,
                "ref_x": ref_x,
                "ref_y": ref_y,
                "recon_x": recon_x,
                "recon_y": recon_y,
                "ref_td_raw": ref_td_raw,
                "recon_td_raw": recon_td_raw,
                "offsets": offsets,
            }
        )
        if first_ref is None:
            first_ref = df_ref
            first_recon = df_recon

    if len(examples) < requested:
        print(f"Warning: collected {len(examples)} accepted sources after skipping {skipped} candidates.")
    if not examples:
        raise RuntimeError("No matched multiple-image examples were found.")
    return examples, first_ref, first_recon


def plot_examples(examples: list[dict[str, object]], output_path: Path, show: bool) -> None:
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.6), constrained_layout=False)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.09, top=0.90, wspace=0.18)

    for source_idx, (ax, ex) in enumerate(zip(axes, examples), start=1):
        df_ref = ex["df_ref"]
        df_recon = ex["df_recon"]
        kappa = np.asarray(df_ref.ka, dtype=float)
        theta_x = np.asarray(df_ref.thetax, dtype=float)
        theta_y = np.asarray(df_ref.thetay, dtype=float)
        finite = kappa[np.isfinite(kappa)]
        vmin, vmax = np.percentile(finite, [2.0, 98.0]) if finite.size else (0.0, 1.0)
        ax.imshow(
            kappa,
            origin="lower",
            extent=[theta_x[0], theta_x[-1], theta_y[0], theta_y[-1]],
            cmap="Greys_r",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        display_critical_lines(ax, df_ref)
        for cl in df_recon.tancl():
            x, y = df_recon.getCritPoints(cl)
            ax.plot(x, y, color="tab:red", lw=0.9, alpha=0.45, ls=":")
        for cl in df_recon.radcl():
            x, y = df_recon.getCritPoints(cl)
            ax.plot(x, y, color="tab:blue", lw=0.8, alpha=0.45, ls=":")

        labels = ex["labels"]
        ref_x = np.asarray(ex["ref_x"], dtype=float)
        ref_y = np.asarray(ex["ref_y"], dtype=float)
        recon_x = np.asarray(ex["recon_x"], dtype=float)
        recon_y = np.asarray(ex["recon_y"], dtype=float)
        offsets = np.asarray(ex["offsets"], dtype=float)

        for lab, x0, y0, x1, y1 in zip(labels, ref_x, ref_y, recon_x, recon_y):
            ax.plot([x0, x1], [y0, y1], color="0.25", lw=1.0, alpha=0.75, zorder=4)
            ax.scatter(x0, y0, s=65, marker="o", facecolor="none", edgecolor="tab:cyan", lw=1.6, zorder=5)
            ax.scatter(x1, y1, s=55, marker="x", color="tab:orange", lw=1.8, zorder=6)
            ax.text(
                x0,
                y0,
                lab,
                color="black",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15),
                zorder=7,
            )

        beta = ex["beta"]
        lines = [
            f"z={float(ex['z']):.3f}",
            rf"$\beta=({beta[0]:.2f},{beta[1]:.2f})$ arcsec",
        ]
        lines.extend([f"{lab}: {off:.3f}\"" for lab, off in zip(labels, offsets)])
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.3),
        )
        ax.set_xlim(float(theta_x[0]), float(theta_x[-1]))
        ax.set_ylim(float(theta_y[0]), float(theta_y[-1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\theta_1$ [arcsec]")
        ax.set_ylabel(r"$\theta_2$ [arcsec]")
        ax.set_title(f"Source {source_idx}: matched images", fontsize=12)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="tab:cyan", markersize=8, label="REF image"),
        plt.Line2D([], [], marker="x", linestyle="None", color="tab:orange", markersize=8, label="RECON image"),
        plt.Line2D([], [], color="tab:red", lw=1.2, label="REF tangential critical"),
        plt.Line2D([], [], color="tab:red", lw=1.0, ls=":", label="RECON tangential critical"),
        plt.Line2D([], [], color="tab:blue", lw=1.0, ls="--", label="REF radial critical"),
        plt.Line2D([], [], color="tab:blue", lw=0.9, ls=":", label="RECON radial critical"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("REF/RECON image-matching validation (A = earliest REF image)", fontsize=14, y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved matching validation figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output, DEFAULT_OUTPUT.name)
    examples, _first_ref, _first_recon = collect_examples(args)
    plot_examples(examples, output_path, bool(args.show))
    for i, ex in enumerate(examples, start=1):
        offset_text = ", ".join(f"{lab}={off:.4f}\"" for lab, off in zip(ex["labels"], ex["offsets"]))
        print(f"Source {i} z={float(ex['z']):.3f}: {offset_text}")


if __name__ == "__main__":
    main()
