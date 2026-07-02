import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path("/tmp/mplcfg").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pyLensLib.deflector import deflector
from pyLensLib.raytracer import raytracer


"""
Example:

python /Users/maxmen3/projects/pyLensLib/Test/test_compare_snapshot_deflection_fft_modes.py \
  --snapshot /Users/maxmen3/projects/pyLensLib/Test/snap_058 \
  --subfind-file /Users/maxmen3/projects/pyLensLib/Test/sub_058.0 \
  --prefix /Users/maxmen3/projects/pyLensLib/Test/artifacts_phase5/snapshot058_deflection_fft_modes
"""


MODE_LABELS = {
    "potential_fft": "legacy potential FFT",
    "deflection_fft": "direct alpha FFT",
    "deflection_fft_adaptive": "adaptive alpha FFT",
}


def parse_parttypes(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def build_raytracer_and_deflector(
    co,
    zl,
    zs,
    mass_map,
    fov_mass_arcsec,
    nray,
    fov_ray_arcsec,
    method,
    alpha_interpolation_order=3,
    low_res_factor=4,
    high_res_kernel_size=21,
):
    kwargs = {
        "zl": zl,
        "zs": zs,
        "fov": fov_mass_arcsec,
        "method": method,
        "alpha_interpolation_order": alpha_interpolation_order,
        "low_res_factor": low_res_factor,
        "high_res_kernel_size": high_res_kernel_size,
        "adaptive_return_high_res": True,
    }
    t0 = time.perf_counter()
    rt = raytracer(
        co,
        mass_map,
        Nray=nray,
        FOVray=fov_ray_arcsec,
        fromfile=False,
        **kwargs,
    )
    elapsed = time.perf_counter() - t0

    df = deflector(co, angx=rt.a1, angy=rt.a2, zl=zl, zs=zs)
    theta = np.linspace(-fov_ray_arcsec / 2.0, fov_ray_arcsec / 2.0, nray)
    df.setGrid(theta=theta, compute_potential=False)
    return rt, df, elapsed


def central_mask(shape, fraction):
    if fraction >= 1.0:
        return np.ones(shape, dtype=bool)
    if fraction <= 0.0:
        raise ValueError("inner fraction must be positive.")
    ny, nx = shape
    y0 = int(round(0.5 * ny * (1.0 - fraction)))
    x0 = int(round(0.5 * nx * (1.0 - fraction)))
    y1 = ny - y0
    x1 = nx - x0
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "rms": np.nan,
            "p95_abs": np.nan,
            "max_abs": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95_abs": float(np.percentile(np.abs(values), 95.0)),
        "max_abs": float(np.max(np.abs(values))),
    }


def map_difference_metrics(candidate, reference, mask):
    diff = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    masked = diff[mask]
    ref_masked = np.asarray(reference, dtype=float)[mask]
    denom = np.sqrt(np.nanmean(ref_masked * ref_masked))
    stats = finite_stats(masked)
    stats["relative_rms"] = float(stats["rms"] / denom) if denom > 0.0 else np.nan
    return stats


def summarize_critical_lines(df):
    try:
        tan = df.tancl()
        rad = df.radcl()
        theta_e = df.thetaEv(clt=tan) if len(tan) > 0 else np.array([])
    except Exception as exc:
        return {
            "error": str(exc),
            "n_tancl": 0,
            "n_radcl": 0,
            "theta_e": [],
        }
    return {
        "n_tancl": int(len(tan)),
        "n_radcl": int(len(rad)),
        "theta_e": [float(x) for x in theta_e],
    }


def compare_modes(results, reference_method, inner_fraction):
    ref = results[reference_method]
    ref_alpha = np.hypot(ref["df"].a1, ref["df"].a2)
    mask = central_mask(ref_alpha.shape, inner_fraction)
    metrics = {}

    for method, payload in results.items():
        df = payload["df"]
        alpha = np.hypot(df.a1, df.a2)
        method_metrics = {
            "label": MODE_LABELS.get(method, method),
            "runtime_sec": float(payload["runtime_sec"]),
            "finite_a1": bool(np.all(np.isfinite(df.a1))),
            "finite_a2": bool(np.all(np.isfinite(df.a2))),
            "finite_kappa": bool(np.all(np.isfinite(df.ka))),
            "critical_lines": summarize_critical_lines(df),
        }
        if method != reference_method:
            method_metrics["delta_vs_reference"] = {
                "a1": map_difference_metrics(df.a1, ref["df"].a1, mask),
                "a2": map_difference_metrics(df.a2, ref["df"].a2, mask),
                "alpha_abs": map_difference_metrics(alpha, ref_alpha, mask),
                "kappa_from_alpha_gradient": map_difference_metrics(df.ka, ref["df"].ka, mask),
                "gamma1_from_alpha_gradient": map_difference_metrics(df.g1, ref["df"].g1, mask),
                "gamma2_from_alpha_gradient": map_difference_metrics(df.g2, ref["df"].g2, mask),
            }
        metrics[method] = method_metrics
    return metrics


def radial_profile(arr, fov_arcsec, nbins=60):
    ny, nx = arr.shape
    x = np.linspace(-fov_arcsec / 2.0, fov_arcsec / 2.0, nx)
    y = np.linspace(-fov_arcsec / 2.0, fov_arcsec / 2.0, ny)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx * xx + yy * yy)
    bins = np.linspace(0.0, rr.max(), nbins + 1)
    rc = 0.5 * (bins[:-1] + bins[1:])
    prof = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (rr >= bins[i]) & (rr < bins[i + 1])
        if np.any(mask):
            prof[i] = np.nanmean(arr[mask])
    return rc, prof


def save_maps_npz(path, results, mass_map):
    payload = {"mass_map": mass_map}
    for method, result in results.items():
        df = result["df"]
        rt = result["rt"]
        key = method.replace("-", "_")
        payload[f"{key}_a1"] = df.a1
        payload[f"{key}_a2"] = df.a2
        payload[f"{key}_alpha_abs"] = np.hypot(df.a1, df.a2)
        payload[f"{key}_kappa_from_alpha_gradient"] = df.ka
        payload[f"{key}_raytracer_potential"] = rt.pot
        payload[f"{key}_input_kappa"] = rt.kappa_map
    np.savez_compressed(path, **payload)


def make_summary_plot(path, results, reference_method, fov_ray_arcsec):
    methods = list(results.keys())
    ref_alpha = np.hypot(
        results[reference_method]["df"].a1,
        results[reference_method]["df"].a2,
    )
    extent = [
        -fov_ray_arcsec / 2.0,
        fov_ray_arcsec / 2.0,
        -fov_ray_arcsec / 2.0,
        fov_ray_arcsec / 2.0,
    ]

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(
        4,
        len(methods),
        height_ratios=[1.0, 1.0, 1.0, 0.85],
        wspace=0.18,
        hspace=0.28,
    )
    vmin = min(
        np.nanpercentile(np.hypot(results[m]["df"].a1, results[m]["df"].a2), 5.0)
        for m in methods
    )
    vmax = max(
        np.nanpercentile(np.hypot(results[m]["df"].a1, results[m]["df"].a2), 99.0)
        for m in methods
    )

    for col, method in enumerate(methods):
        df = results[method]["df"]
        alpha = np.hypot(df.a1, df.a2)
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(alpha, origin="lower", extent=extent, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(MODE_LABELS.get(method, method))
        ax.set_aspect("equal")
        if col == 0:
            ax.set_ylabel(r"$\theta_2$ [arcsec]")
        ax.set_xlabel(r"$\theta_1$ [arcsec]")

        ax = fig.add_subplot(gs[1, col])
        if method == reference_method:
            diff = np.zeros_like(alpha)
        else:
            diff = alpha - ref_alpha
        dv = np.nanpercentile(np.abs(diff), 99.0)
        if dv == 0.0 or not np.isfinite(dv):
            dv = 1.0
        ax.imshow(diff, origin="lower", extent=extent, cmap="coolwarm", vmin=-dv, vmax=dv)
        ax.set_title(r"$\Delta |\alpha|$ vs legacy")
        ax.set_aspect("equal")
        if col == 0:
            ax.set_ylabel(r"$\theta_2$ [arcsec]")
        ax.set_xlabel(r"$\theta_1$ [arcsec]")

        ax = fig.add_subplot(gs[2, col])
        kappa = df.ka
        kvmax = np.nanpercentile(np.abs(kappa), 99.0)
        ax.imshow(kappa, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=kvmax)
        ax.set_title(r"$\kappa$ from $\nabla\alpha$")
        ax.set_aspect("equal")
        if col == 0:
            ax.set_ylabel(r"$\theta_2$ [arcsec]")
        ax.set_xlabel(r"$\theta_1$ [arcsec]")

    cax = fig.add_axes([0.92, 0.735, 0.015, 0.18])
    fig.colorbar(im, cax=cax, label=r"$|\alpha|$ [arcsec]")

    axp = fig.add_subplot(gs[3, :])
    for method in methods:
        df = results[method]["df"]
        alpha = np.hypot(df.a1, df.a2)
        r, prof = radial_profile(alpha, fov_ray_arcsec)
        axp.plot(r, prof, lw=1.8, label=MODE_LABELS.get(method, method))
    axp.set_xlabel("R [arcsec]")
    axp.set_ylabel(r"azimuthal mean $|\alpha|$ [arcsec]")
    axp.grid(alpha=0.25)
    axp.legend(frameon=False, ncol=3)

    fig.suptitle("Snapshot deflection maps: legacy potential FFT vs direct/adaptive alpha FFT")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_critical_lines_plot(path, results, fov_ray_arcsec, zs):
    methods = list(results.keys())
    extent = [
        -fov_ray_arcsec / 2.0,
        fov_ray_arcsec / 2.0,
        -fov_ray_arcsec / 2.0,
        fov_ray_arcsec / 2.0,
    ]

    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 5.2), sharex=True, sharey=True)
    if len(methods) == 1:
        axes = [axes]

    all_kappa = [results[method]["df"].ka for method in methods]
    vmax = max(np.nanpercentile(np.abs(kappa), 99.0) for kappa in all_kappa)

    for ax, method in zip(axes, methods):
        df = results[method]["df"]
        ax.imshow(df.ka, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=vmax)

        tan = df.tancl()
        rad = df.radcl()
        for i, crit in enumerate(tan):
            x, y = df.getCritPoints(crit, pixel_units=False)
            label = "tangential" if i == 0 else None
            ax.plot(x, y, color="white", lw=1.4, label=label)
        for i, crit in enumerate(rad):
            x, y = df.getCritPoints(crit, pixel_units=False)
            label = "radial" if i == 0 else None
            ax.plot(x, y, color="cyan", lw=1.1, ls="--", label=label)

        ax.set_title(f"{MODE_LABELS.get(method, method)}\nT={len(tan)} R={len(rad)}")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\theta_1$ [arcsec]")
        ax.grid(alpha=0.15)

    axes[0].set_ylabel(r"$\theta_2$ [arcsec]")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"Snapshot critical lines at z_s={zs:g}: tangential white, radial cyan")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_json_summary(path, args, mass_meta, metrics):
    payload = {
        "snapshot": str(args.snapshot),
        "subfind_file": str(args.subfind_file),
        "parttypes": parse_parttypes(args.parttypes),
        "zs": float(args.zs),
        "fov_arcsec": float(args.fov_arcsec),
        "fov_ray_arcsec": float(args.fov_ray_arcsec),
        "npix_map": int(args.npix_map),
        "nray": int(args.nray),
        "zhalf_mpc": float(args.zhalf_mpc),
        "inner_fraction": float(args.inner_fraction),
        "alpha_interpolation_order": int(args.alpha_interpolation_order),
        "adaptive": {
            "low_res_factor": int(args.low_res_factor),
            "high_res_kernel_size": int(args.high_res_kernel_size),
            "adaptive_return_high_res": True,
        },
        "mass_map_meta": mass_meta,
        "metrics": metrics,
    }
    with open(path, "w") as fobj:
        json.dump(payload, fobj, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare snapshot deflection maps generated with the legacy "
            "potential FFT raytracer and the direct/adaptive alpha FFT modes."
        )
    )
    parser.add_argument("--snapshot", type=Path, default=Path("/Users/maxmen3/projects/pyLensLib/Test/snap_058"))
    parser.add_argument("--subfind-file", type=Path, default=Path("/Users/maxmen3/projects/pyLensLib/Test/sub_058.0"))
    parser.add_argument("--prefix", type=Path, default=Path("/Users/maxmen3/projects/pyLensLib/Test/artifacts_phase5/snapshot058_deflection_fft_modes"))
    parser.add_argument("--parttypes", type=str, default="0,1,4")
    parser.add_argument("--fov-arcsec", type=float, default=200.0)
    parser.add_argument("--fov-ray-arcsec", type=float, default=None)
    parser.add_argument("--npix-map", type=int, default=256)
    parser.add_argument("--nray", type=int, default=256)
    parser.add_argument("--zhalf-mpc", type=float, default=10.0)
    parser.add_argument("--zs", type=float, default=3.0)
    parser.add_argument("--nb", type=int, default=32)
    parser.add_argument("--min-hsml", type=float, default=1e-6)
    parser.add_argument("--max-hsml", type=float, default=0.1)
    parser.add_argument("--low-res-factor", type=int, default=4)
    parser.add_argument("--high-res-kernel-size", type=int, default=21)
    parser.add_argument("--alpha-interpolation-order", type=int, default=3)
    parser.add_argument("--inner-fraction", type=float, default=0.8)
    parser.add_argument("--save-npz", action="store_true")
    args = parser.parse_args()

    from test_compare_snapshot_crosssections_py_jl import build_total_mass_map

    fov_ray_arcsec = args.fov_arcsec if args.fov_ray_arcsec is None else args.fov_ray_arcsec
    args.fov_ray_arcsec = fov_ray_arcsec
    prefix = args.prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    cl_ref, mass_map, mass_meta = build_total_mass_map(
        snapshot=str(args.snapshot),
        subfind_file=str(args.subfind_file),
        parttypes=parse_parttypes(args.parttypes),
        fov_arcsec=args.fov_arcsec,
        npix_map=args.npix_map,
        zhalf_mpc=args.zhalf_mpc,
        nb=args.nb,
        min_hsml=args.min_hsml,
        max_hsml=args.max_hsml,
    )

    methods = ["potential_fft", "deflection_fft", "deflection_fft_adaptive"]
    results = {}
    for method in methods:
        rt, df, elapsed = build_raytracer_and_deflector(
            co=cl_ref.co,
            zl=cl_ref.zl,
            zs=args.zs,
            mass_map=mass_map,
            fov_mass_arcsec=args.fov_arcsec,
            nray=args.nray,
            fov_ray_arcsec=fov_ray_arcsec,
            method=method,
            alpha_interpolation_order=args.alpha_interpolation_order,
            low_res_factor=args.low_res_factor,
            high_res_kernel_size=args.high_res_kernel_size,
        )
        results[method] = {
            "rt": rt,
            "df": df,
            "runtime_sec": elapsed,
        }
        print(f"{method}: {elapsed:.3f} s")

    metrics = compare_modes(
        results,
        reference_method="potential_fft",
        inner_fraction=args.inner_fraction,
    )

    json_path = prefix.with_suffix(".json")
    png_path = prefix.with_suffix(".png")
    crit_png_path = prefix.with_name(prefix.name + "_critical_lines").with_suffix(".png")
    write_json_summary(json_path, args, mass_meta, metrics)
    make_summary_plot(png_path, results, "potential_fft", fov_ray_arcsec)
    make_critical_lines_plot(crit_png_path, results, fov_ray_arcsec, args.zs)

    print(json_path)
    print(png_path)
    print(crit_png_path)

    if args.save_npz:
        npz_path = prefix.with_suffix(".npz")
        save_maps_npz(npz_path, results, mass_map)
        print(npz_path)


if __name__ == "__main__":
    main()
