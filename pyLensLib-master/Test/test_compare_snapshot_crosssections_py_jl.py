import argparse
import csv
import json
import subprocess
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pyLensLib.cluster import cluster
from pyLensLib.deflector import deflector
from pyLensLib.raytracer import raytracer
from pyLensLib.subfind import subfind


"""
python /Users/maxmen3/projects/pyLensLib/Test/test_compare_snapshot_crosssections_py_jl.py \
  --snapshot /Users/maxmen3/projects/pyLensLib/Test/snap_058 \
  --subfind-file /Users/maxmen3/projects/pyLensLib/Test/sub_058.0 \
  --prefix /Users/maxmen3/projects/pyLensLib/Test/artifacts_phase5/snapshot058_crosssections
"""


JULIA_BIN = "/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia"
JL_HELPER = "/Users/maxmen3/projects/jlLensLib/test/phase5_snapshot_crosssections_geometry.jl"
JL_PROJECT = "/Users/maxmen3/projects/jlLensLib"


def parse_parttypes(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def load_center_from_subfind(subfind_file, index=0):
    sf = subfind(subfind_file)
    xc, yc, zc = sf.gpos[index]
    return float(xc), float(yc), float(zc)


def build_total_mass_map(
    snapshot,
    subfind_file,
    parttypes,
    fov_arcsec,
    npix_map,
    zhalf_mpc,
    nb=32,
    min_hsml=1e-6,
    max_hsml=0.1,
    subfind_index=0,
    hdf5=False,
    fsample=0.0,
):
    xc, yc, zc = load_center_from_subfind(subfind_file, index=subfind_index)
    cl_ref = cluster(
        snapshot=snapshot,
        parttype=[parttypes[0]],
        xc=xc,
        yc=yc,
        zc=zc,
        hdf5=hdf5,
        fsample=fsample,
    )

    dl = cl_ref.co.angular_diameter_distance(cl_ref.zl).value
    fov_mpc = np.deg2rad(fov_arcsec / 3600.0) * dl
    bounds = dict(
        xmin=-fov_mpc / 2.0,
        xmax=fov_mpc / 2.0,
        ymin=-fov_mpc / 2.0,
        ymax=fov_mpc / 2.0,
    )
    mm_kwargs = dict(
        npix=npix_map,
        xmin=bounds["xmin"],
        xmax=bounds["xmax"],
        ymin=bounds["ymin"],
        ymax=bounds["ymax"],
        zmin=-zhalf_mpc,
        zmax=+zhalf_mpc,
        nb=nb,
        min_hsml=min_hsml,
        max_hsml=max_hsml,
    )

    img = np.zeros((npix_map, npix_map), dtype=np.float64)
    per_type = []
    for parttype in parttypes:
        cl_i = cluster(
            snapshot=snapshot,
            parttype=[parttype],
            xc=xc,
            yc=yc,
            zc=zc,
            hdf5=hdf5,
            fsample=fsample,
        )
        this = cl_i.massMapSPH_internal(**mm_kwargs)
        img += this
        per_type.append(
            {
                "parttype": int(parttype),
                "projected_mass_msun": float(this.sum()),
            }
        )

    meta = {
        "center_mpc": [float(xc), float(yc), float(zc)],
        "fov_mpc": float(fov_mpc),
        "bounds_mpc": {k: float(v) for k, v in bounds.items()},
        "per_parttype": per_type,
    }
    return cl_ref, img, meta


def build_deflector_from_mass_map(co, zl, zs, mass_map, fov_mass_arcsec, nray, fov_ray_arcsec):
    kwargs = {"zl": zl, "zs": zs, "fov": fov_mass_arcsec}
    rt = raytracer(co, mass_map, Nray=nray, FOVray=fov_ray_arcsec, fromfile=False, **kwargs)
    df = deflector(co, angx=rt.a1, angy=rt.a2, zl=zl, zs=zs)
    theta = np.linspace(-fov_ray_arcsec / 2.0, fov_ray_arcsec / 2.0, nray)
    df.setGrid(theta=theta, compute_potential=False)
    return df


def julia_metrics_from_h5(h5_path, co, zl, fov_arcsec, ggsl_minsize=0.5, ggsl_maxsize=5.0, ggsl_dmax=200.0):
    rows = []
    with h5py.File(h5_path, "r") as h5:
        zs_grid = np.array(h5["zs_grid"])
        for iz, zs in enumerate(zs_grid, start=1):
            gz = h5[f"z{iz}"]
            a1 = np.array(gz["a1"])
            a2 = np.array(gz["a2"])
            theta = np.linspace(-fov_arcsec / 2.0, fov_arcsec / 2.0, a1.shape[0])
            df = deflector(co, angx=a1, angy=a2, zl=zl, zs=float(zs))
            df.setGrid(theta=theta, compute_potential=False)
            tl = df.tancl()
            rl = df.radcl()
            _, _, sp_area = df.fovSP()
            ggsl_area = float(df.ggslCrossSection(clt=tl, minsize=ggsl_minsize, maxsize=ggsl_maxsize, dmax=ggsl_dmax))
            mi_area = float(df.multImaCrossSection())

            rows.append(
                {
                    "zs": float(zs),
                    "source_plane_area": float(sp_area),
                    "ggsl_cross_section": float(ggsl_area),
                    "multima_cross_section": float(mi_area),
                    "ggsl_probability": float(ggsl_area / sp_area) if sp_area > 0 else np.nan,
                    "multima_probability": float(mi_area / sp_area) if sp_area > 0 else np.nan,
                    "n_tancl": int(len(tl)),
                    "n_radcl": int(len(rl)),
                }
            )
    return rows

def write_massmap_input_h5(path, mass_map, co, zl, fov_arcsec):
    with h5py.File(path, "w") as h5:
        h5["mass_map"] = mass_map
        h5.attrs["redshift"] = float(zl)
        h5.attrs["omega_m"] = float(co.Om0)
        h5.attrs["hubble"] = float(co.H0.value / 100.0)
        h5.attrs["fov_arcsec"] = float(fov_arcsec)


def run_julia_helper(args, out_h5, massmap_h5):
    cmd = [
        JULIA_BIN,
        f"--project={JL_PROJECT}",
        JL_HELPER,
        "--massmap-h5",
        str(massmap_h5),
        "--out",
        str(out_h5),
        "--fov-arcsec",
        str(args.fov_arcsec),
        "--npix-map",
        str(args.npix_map),
        "--nray",
        str(args.nray),
        "--halo-index",
        "1",
        "--zhalf-mpc",
        str(args.zhalf_mpc),
        "--zmin",
        str(args.zmin),
        "--zmax",
        str(args.zmax),
        "--nz",
        str(args.nz),
        "--parttypes",
        args.parttypes,
        "--backend",
        args.jl_backend,
    ]
    subprocess.run(cmd, check=True)


def py_metrics_from_mass_map(cl_ref, mass_map, zs_grid, fov_arcsec, nray, ggsl_minsize=0.5, ggsl_maxsize=5.0, ggsl_dmax=200.0):
    rows = []
    for zs in zs_grid:
        df = build_deflector_from_mass_map(
            cl_ref.co,
            zl=cl_ref.zl,
            zs=float(zs),
            mass_map=mass_map,
            fov_mass_arcsec=fov_arcsec,
            nray=nray,
            fov_ray_arcsec=fov_arcsec,
        )
        tl = df.tancl()
        rl = df.radcl()
        _, _, sp_area = df.fovSP()
        rows.append(
            {
                "zs": float(zs),
                "source_plane_area": float(sp_area),
                "ggsl_cross_section": float(df.ggslCrossSection(clt=tl, minsize=ggsl_minsize, maxsize=ggsl_maxsize, dmax=ggsl_dmax)),
                "multima_cross_section": float(df.multImaCrossSection()),
                "ggsl_probability": float(df.ggslCrossSection(clt=tl, minsize=ggsl_minsize, maxsize=ggsl_maxsize, dmax=ggsl_dmax) / sp_area) if sp_area > 0 else np.nan,
                "multima_probability": float(df.multImaCrossSection() / sp_area) if sp_area > 0 else np.nan,
                "n_tancl": int(len(tl)),
                "n_radcl": int(len(rl)),
            }
        )
    return rows


def write_summary_csv(path, py_rows, jl_rows):
    fieldnames = [
        "zs",
        "py_source_plane_area",
        "jl_source_plane_area",
        "py_ggsl_cross_section",
        "jl_ggsl_cross_section",
        "py_multima_cross_section",
        "jl_multima_cross_section",
        "py_ggsl_probability",
        "jl_ggsl_probability",
        "py_multima_probability",
        "jl_multima_probability",
        "py_n_tancl",
        "jl_n_tancl",
        "py_n_radcl",
        "jl_n_radcl",
    ]
    with open(path, "w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for py_row, jl_row in zip(py_rows, jl_rows):
            writer.writerow(
                {
                    "zs": py_row["zs"],
                    "py_source_plane_area": py_row["source_plane_area"],
                    "jl_source_plane_area": jl_row["source_plane_area"],
                    "py_ggsl_cross_section": py_row["ggsl_cross_section"],
                    "jl_ggsl_cross_section": jl_row["ggsl_cross_section"],
                    "py_multima_cross_section": py_row["multima_cross_section"],
                    "jl_multima_cross_section": jl_row["multima_cross_section"],
                    "py_ggsl_probability": py_row["ggsl_probability"],
                    "jl_ggsl_probability": jl_row["ggsl_probability"],
                    "py_multima_probability": py_row["multima_probability"],
                    "jl_multima_probability": jl_row["multima_probability"],
                    "py_n_tancl": py_row["n_tancl"],
                    "jl_n_tancl": jl_row["n_tancl"],
                    "py_n_radcl": py_row["n_radcl"],
                    "jl_n_radcl": jl_row["n_radcl"],
                }
            )


def make_summary_plot(path, py_rows, jl_rows):
    zs = np.array([row["zs"] for row in py_rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    panels = [
        ("ggsl_cross_section", "GGSL Cross Section [arcsec$^2$]"),
        ("multima_cross_section", "Multiple-Image Cross Section [arcsec$^2$]"),
        ("ggsl_probability", "GGSL Probability"),
        ("multima_probability", "Multiple-Image Probability"),
    ]
    for ax, (key, ylabel) in zip(axes.flat, panels):
        py_vals = np.array([row[key] for row in py_rows], dtype=float)
        jl_vals = np.array([row[key] for row in jl_rows], dtype=float)
        ax.plot(zs, py_vals, marker="o", lw=1.8, alpha=0.5, label="pyLensLib")
        ax.plot(zs, jl_vals, marker="s", lw=1.8, alpha=0.5, label="jlLensLib")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Source redshift")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Snapshot 058 Cross Sections: pyLensLib vs jlLensLib")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_json(path, args, mass_meta, py_rows, jl_rows, julia_h5):
    payload = {
        "snapshot": str(args.snapshot),
        "subfind_file": str(args.subfind_file),
        "parttypes": parse_parttypes(args.parttypes),
        "fov_arcsec": float(args.fov_arcsec),
        "npix_map": int(args.npix_map),
        "nray": int(args.nray),
        "zhalf_mpc": float(args.zhalf_mpc),
        "zs_grid": [float(row["zs"]) for row in py_rows],
        "mass_map_meta": mass_meta,
        "pyLensLib": py_rows,
        "jlLensLib": jl_rows,
        "jl_geometry_h5": str(julia_h5),
    }
    with open(path, "w") as fobj:
        json.dump(payload, fobj, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compare snapshot-based GGSL and multiple-image cross sections between pyLensLib and jlLensLib.")
    parser.add_argument("--snapshot", type=Path, default=Path("/Users/maxmen3/projects/pyLensLib/Test/snap_058"))
    parser.add_argument("--subfind-file", type=Path, default=Path("/Users/maxmen3/projects/pyLensLib/Test/sub_058.0"))
    parser.add_argument("--prefix", type=Path, default=Path("/Users/maxmen3/projects/pyLensLib/Test/artifacts_phase5/snapshot058_crosssections"))
    parser.add_argument("--parttypes", type=str, default="0,1,4")
    parser.add_argument("--fov-arcsec", type=float, default=200.0)
    parser.add_argument("--npix-map", type=int, default=256)
    parser.add_argument("--nray", type=int, default=256)
    parser.add_argument("--zhalf-mpc", type=float, default=10.0)
    parser.add_argument("--nb", type=int, default=32)
    parser.add_argument("--min-hsml", type=float, default=1e-6)
    parser.add_argument("--max-hsml", type=float, default=0.1)
    parser.add_argument("--zmin", type=float, default=1.0)
    parser.add_argument("--zmax", type=float, default=6.0)
    parser.add_argument("--nz", type=int, default=10)
    parser.add_argument("--jl-backend", type=str, default="auto")
    args = parser.parse_args()

    prefix = args.prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    parttypes = parse_parttypes(args.parttypes)

    cl_ref, mass_map, mass_meta = build_total_mass_map(
        snapshot=str(args.snapshot),
        subfind_file=str(args.subfind_file),
        parttypes=parttypes,
        fov_arcsec=args.fov_arcsec,
        npix_map=args.npix_map,
        zhalf_mpc=args.zhalf_mpc,
        nb=args.nb,
        min_hsml=args.min_hsml,
        max_hsml=args.max_hsml,
    )
    zs_grid = np.linspace(args.zmin, args.zmax, args.nz)
    py_rows = py_metrics_from_mass_map(cl_ref, mass_map, zs_grid, args.fov_arcsec, args.nray)

    julia_h5 = prefix.with_suffix(".jl_geometry.h5")
    massmap_h5 = prefix.with_suffix(".massmap_input.h5")
    write_massmap_input_h5(massmap_h5, mass_map, cl_ref.co, cl_ref.zl, args.fov_arcsec)
    run_julia_helper(args, julia_h5, massmap_h5)
    jl_rows = julia_metrics_from_h5(julia_h5, cl_ref.co, cl_ref.zl, args.fov_arcsec)

    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    fig_path = prefix.with_suffix(".png")
    write_summary_csv(csv_path, py_rows, jl_rows)
    write_summary_json(json_path, args, mass_meta, py_rows, jl_rows, julia_h5)
    make_summary_plot(fig_path, py_rows, jl_rows)

    print(csv_path)
    print(json_path)
    print(fig_path)


if __name__ == "__main__":
    main()
