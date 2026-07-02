import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pyLensLib.deflector import deflector

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from test_compare_snapshot_crosssections_py_jl import build_total_mass_map, build_deflector_from_mass_map
from test_compare_snapshot_crosssections_py_jl import write_massmap_input_h5

JULIA_BIN = "/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia"
JL_HELPER = "/Users/maxmen3/projects/jlLensLib/test/phase5_snapshot_crosssections_geometry.jl"
JL_PROJECT = "/Users/maxmen3/projects/jlLensLib"


def run_julia_helper(massmap_h5, out_h5, fov_arcsec, npix_map, nray, zhalf_mpc, parttypes, backend, zs):
    cmd = [
        JULIA_BIN,
        f"--project={JL_PROJECT}",
        JL_HELPER,
        "--massmap-h5",
        str(massmap_h5),
        "--out",
        str(out_h5),
        "--fov-arcsec",
        str(fov_arcsec),
        "--npix-map",
        str(npix_map),
        "--nray",
        str(nray),
        "--zhalf-mpc",
        str(zhalf_mpc),
        "--zmin",
        str(zs),
        "--zmax",
        str(zs),
        "--nz",
        "1",
        "--parttypes",
        parttypes,
        "--backend",
        backend,
    ]
    subprocess.run(cmd, check=True)


def _points_group_to_xy(group):
    pts = np.array(group["points_arcsec"])
    if pts.ndim == 2 and pts.shape[1] == 2:
        return pts[:, 0], pts[:, 1]
    if pts.ndim == 2 and pts.shape[0] == 2:
        return pts[0], pts[1]
    raise ValueError(f"Unexpected points shape {pts.shape}")


def load_jl_criticals(h5_path):
    tangential = []
    radial = []
    with h5py.File(h5_path, "r") as h5:
        g = h5["z1"]
        kappa = np.array(g["kappa"])
        a1 = np.array(g["a1"])
        a2 = np.array(g["a2"])
        gt = g["tangential"]
        gr = g["radial"]
        for i in range(1, int(gt.attrs["n"]) + 1):
            tangential.append(_points_group_to_xy(gt[f"critical_{i}"]))
        for i in range(1, int(gr.attrs["n"]) + 1):
            radial.append(_points_group_to_xy(gr[f"critical_{i}"]))
    return kappa, a1, a2, tangential, radial


def radial_profile(kappa, fov_arcsec, nbins=60):
    arr = np.asarray(kappa, dtype=float)
    ny, nx = arr.shape
    x = np.linspace(-fov_arcsec / 2.0, fov_arcsec / 2.0, nx)
    y = np.linspace(-fov_arcsec / 2.0, fov_arcsec / 2.0, ny)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    rmax = rr.max()
    bins = np.linspace(0.0, rmax, nbins + 1)
    rc = 0.5 * (bins[:-1] + bins[1:])
    prof = np.full(nbins, np.nan, dtype=float)
    for i in range(nbins):
        sel = (rr >= bins[i]) & (rr < bins[i + 1])
        if np.any(sel):
            prof[i] = float(np.nanmean(arr[sel]))
    return rc, prof


def plot_py_critical(ax, criticals, pixel_scale, nray, color, ls="-", lw=1.4, alpha=1.0, label=None):
    first = True
    for cl in criticals:
        pts = np.asarray(list(cl.points), dtype=float)
        x = (pts[:, 0] - nray / 2.0) * pixel_scale
        y = (pts[:, 1] - nray / 2.0) * pixel_scale
        ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=alpha, label=label if first else None)
        first = False


def plot_jl_critical(ax, criticals, color, ls="-", lw=1.4, alpha=1.0, label=None):
    first = True
    for x, y in criticals:
        ax.plot(y, x, color=color, lw=lw, ls=ls, alpha=alpha, label=label if first else None)
        first = False


def main():
    parser = argparse.ArgumentParser(description="Plot pyLensLib vs jlLensLib critical lines for snap_058 at a given source redshift.")
    parser.add_argument("--snapshot", type=Path, default=Path("/Users/maxmen3/projects/pyLensLib/Test/snap_058"))
    parser.add_argument("--subfind-file", type=Path, default=Path("/Users/maxmen3/projects/pyLensLib/Test/sub_058.0"))
    parser.add_argument("--prefix", type=Path, default=Path("/Users/maxmen3/projects/pyLensLib/Test/artifacts_phase5/snapshot058_critical_lines_zs3"))
    parser.add_argument("--parttypes", type=str, default="0,1,4")
    parser.add_argument("--fov-arcsec", type=float, default=200.0)
    parser.add_argument("--npix-map", type=int, default=256)
    parser.add_argument("--nray", type=int, default=256)
    parser.add_argument("--zhalf-mpc", type=float, default=10.0)
    parser.add_argument("--nb", type=int, default=32)
    parser.add_argument("--min-hsml", type=float, default=1e-6)
    parser.add_argument("--max-hsml", type=float, default=0.1)
    parser.add_argument("--zs", type=float, default=3.0)
    parser.add_argument("--jl-backend", type=str, default="auto")
    args = parser.parse_args()

    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    cl_ref, mass_map, _ = build_total_mass_map(
        snapshot=str(args.snapshot),
        subfind_file=str(args.subfind_file),
        parttypes=[int(x.strip()) for x in args.parttypes.split(",") if x.strip()],
        fov_arcsec=args.fov_arcsec,
        npix_map=args.npix_map,
        zhalf_mpc=args.zhalf_mpc,
        nb=args.nb,
        min_hsml=args.min_hsml,
        max_hsml=args.max_hsml,
    )

    df_py = build_deflector_from_mass_map(cl_ref.co, cl_ref.zl, args.zs, mass_map, args.fov_arcsec, args.nray, args.fov_arcsec)
    tl_py = df_py.tancl()
    rl_py = df_py.radcl()
    kappa_py = df_py.ka

    massmap_h5 = args.prefix.with_suffix(".massmap_input.h5")
    write_massmap_input_h5(massmap_h5, mass_map, cl_ref.co, cl_ref.zl, args.fov_arcsec)
    jl_h5 = args.prefix.with_suffix(".jl_geometry.h5")
    run_julia_helper(massmap_h5, jl_h5, args.fov_arcsec, args.npix_map, args.nray, args.zhalf_mpc, args.parttypes, args.jl_backend, args.zs)
    kappa_jl_native, a1_jl, a2_jl, tl_jl_native, rl_jl_native = load_jl_criticals(jl_h5)
    theta = np.linspace(-args.fov_arcsec / 2.0, args.fov_arcsec / 2.0, a1_jl.shape[0])
    df_jl = deflector(cl_ref.co, angx=a1_jl, angy=a2_jl, zl=cl_ref.zl, zs=args.zs)
    df_jl.setGrid(theta=theta, compute_potential=False)
    tl_jl = df_jl.tancl()
    rl_jl = df_jl.radcl()
    kappa_jl_plot = df_jl.ka

    extent = [-args.fov_arcsec / 2.0, args.fov_arcsec / 2.0, -args.fov_arcsec / 2.0, args.fov_arcsec / 2.0]
    vmin = min(np.nanpercentile(kappa_py, 5), np.nanpercentile(kappa_jl_plot, 5))
    vmax = max(np.nanpercentile(kappa_py, 99), np.nanpercentile(kappa_jl_plot, 99))

    fig = plt.figure(figsize=(12.8, 9.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.05], height_ratios=[1.0, 0.72], wspace=0.12, hspace=0.22)
    ax_py = fig.add_subplot(gs[0, 0])
    ax_jl = fig.add_subplot(gs[0, 1], sharex=ax_py, sharey=ax_py)
    cax = fig.add_subplot(gs[0, 2])
    ax_prof = fig.add_subplot(gs[1, 0:2])

    im = ax_py.imshow(kappa_py, origin="lower", extent=extent, cmap="Greys", vmin=vmin, vmax=vmax)
    ax_jl.imshow(kappa_jl_plot, origin="lower", extent=extent, cmap="Greys", vmin=vmin, vmax=vmax)

    plot_py_critical(ax_py, tl_py, df_py.pixel_scale, args.nray, color="tab:red", ls="-", lw=1.4, label="py tangential")
    plot_py_critical(ax_py, rl_py, df_py.pixel_scale, args.nray, color="tab:blue", ls="--", lw=1.1, label="py radial")
    plot_py_critical(ax_py, tl_jl, df_jl.pixel_scale, args.nray, color="gold", ls="-", lw=1.4, alpha=0.95, label="jl tangential")
    plot_py_critical(ax_py, rl_jl, df_jl.pixel_scale, args.nray, color="cyan", ls="--", lw=1.1, alpha=0.95, label="jl radial")

    plot_py_critical(ax_jl, tl_py, df_py.pixel_scale, args.nray, color="tab:red", ls="-", lw=1.4, label="py tangential")
    plot_py_critical(ax_jl, rl_py, df_py.pixel_scale, args.nray, color="tab:blue", ls="--", lw=1.1, label="py radial")
    plot_py_critical(ax_jl, tl_jl, df_jl.pixel_scale, args.nray, color="gold", ls="-", lw=1.4, alpha=0.95, label="jl tangential")
    plot_py_critical(ax_jl, rl_jl, df_jl.pixel_scale, args.nray, color="cyan", ls="--", lw=1.1, alpha=0.95, label="jl radial")

    for ax, title in [(ax_py, "pyLensLib κ map"), (ax_jl, "jlLensLib κ map")]:
        ax.set_title(f"{title} (zs={args.zs:.1f})")
        ax.set_xlabel(r"$\theta_1$ [arcsec]")
        ax.set_aspect("equal")
    ax_py.set_ylabel(r"$\theta_2$ [arcsec]")
    ax_py.legend(frameon=False, fontsize=9, loc="upper right")
    fig.colorbar(im, cax=cax, label=r"$\kappa$")

    r_py, p_py = radial_profile(kappa_py, args.fov_arcsec)
    r_jl, p_jl = radial_profile(kappa_jl_plot, args.fov_arcsec)
    mpy = np.isfinite(p_py) & (p_py > 0.0)
    mjl = np.isfinite(p_jl) & (p_jl > 0.0)
    ax_prof.plot(r_py[mpy], p_py[mpy], color="tab:red", lw=1.8, label="pyLensLib")
    ax_prof.plot(r_jl[mjl], p_jl[mjl], color="goldenrod", lw=1.8, label="jlLensLib")
    ax_prof.set_xlabel(r"$R$ [arcsec]")
    ax_prof.set_ylabel(r"$\langle \kappa \rangle$")
    ax_prof.set_title("Azimuthally Averaged Convergence Profiles")
    ax_prof.set_yscale("log")
    ax_prof.grid(alpha=0.25)
    ax_prof.legend(frameon=False)

    out_png = args.prefix.with_suffix(".png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(out_png)
    print(jl_h5)


if __name__ == "__main__":
    main()
