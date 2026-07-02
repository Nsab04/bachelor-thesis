import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM

from pyLensLib.cluster import cluster
from pyLensLib.deflector import deflector
from pyLensLib.piemd import piemd
from pyLensLib.pointsrc import pointsrc
from pyLensLib.raytracer import raytracer
from pyLensLib.samplers import sample2Dimage
from pyLensLib.subfind import subfind
import pyLensLib.cluster as cluster_module

"""
python test_compare_massMapSPH_internal.py \
  --mode snapshot \
  --snapshot /Users/maxmen3/Downloads/snap_058 \
  --subfind-file /Users/maxmen3/Downloads/sub_058.0 \
  --subfind-index 0 \
  --parttypes 0,1,4 \
  --fov-arcsec 220 \
  --fov-ray-arcsec 180 \
  --npix-map 1024 \
  --nray 1024 \
  --prefix /Users/maxmen3/projects/pyLensLib/Test/sph_cmp_snap_nb40 \
  --max-hsml 0.05 \
  --min-hsml 1e-6 \
  --nb-old 40 \
  --nb-tmp 15
"""


def sigma_crit_msun_mpc2(co, zl, zs):
    dl = co.angular_diameter_distance(zl)
    ds = co.angular_diameter_distance(zs)
    dls = co.angular_diameter_distance_z1z2(zl, zs)
    c2_over_g = (c**2 / G).to(u.Msun / u.Mpc)
    return (c2_over_g / (4.0 * np.pi) * ds / (dl * dls)).value


def synthetic_subhalo_specs():
    return [
        dict(x1=26.0, x2=9.0, sigma0=250.0, q=0.85, pa=10.0, theta_c=0.005, theta_t=20.0),
        dict(x1=-28.0, x2=-12.0, sigma0=210.0, q=0.78, pa=55.0, theta_c=0.005, theta_t=18.0),
        dict(x1=14.0, x2=-31.0, sigma0=190.0, q=0.80, pa=20.0, theta_c=0.005, theta_t=16.0),
        dict(x1=-34.0, x2=22.0, sigma0=175.0, q=0.90, pa=5.0, theta_c=0.005, theta_t=14.0),
        dict(x1=44.0, x2=-18.0, sigma0=165.0, q=0.74, pa=75.0, theta_c=0.004, theta_t=13.0),
        dict(x1=-48.0, x2=2.0, sigma0=155.0, q=0.82, pa=35.0, theta_c=0.004, theta_t=12.0),
        dict(x1=5.0, x2=42.0, sigma0=145.0, q=0.88, pa=115.0, theta_c=0.004, theta_t=11.0),
        dict(x1=-12.0, x2=-49.0, sigma0=135.0, q=0.70, pa=140.0, theta_c=0.004, theta_t=10.0),
        dict(x1=58.0, x2=28.0, sigma0=125.0, q=0.92, pa=25.0, theta_c=0.003, theta_t=9.0),
        dict(x1=-61.0, x2=-31.0, sigma0=118.0, q=0.76, pa=100.0, theta_c=0.003, theta_t=8.5),
        dict(x1=31.0, x2=52.0, sigma0=110.0, q=0.84, pa=60.0, theta_c=0.003, theta_t=8.0),
        dict(x1=-39.0, x2=55.0, sigma0=105.0, q=0.79, pa=130.0, theta_c=0.003, theta_t=7.5),
        dict(x1=67.0, x2=-46.0, sigma0=198.0, q=0.86, pa=15.0, theta_c=0.002, theta_t=17.0),
        dict(x1=-71.0, x2=37.0, sigma0=192.0, q=0.81, pa=95.0, theta_c=0.002, theta_t=16.5),
        dict(x1=18.0, x2=69.0, sigma0=188.0, q=0.73, pa=45.0, theta_c=0.002, theta_t=16.0),
        dict(x1=-7.0, x2=-73.0, sigma0=182.0, q=0.89, pa=160.0, theta_c=0.002, theta_t=15.5),
    ]


def build_synthetic_lens_components(co, zl, zs, fov_arcsec, npix_model):
    theta = np.linspace(-fov_arcsec / 2.0, fov_arcsec / 2.0, npix_model)

    main = piemd(
        co,
        zl=zl,
        zs=zs,
        sigma0=1400.0,
        q=0.72,
        pa=25.0,
        theta_c=2.5,
        theta_t=520.0,
        x1=0.0,
        x2=0.0,
    )
    main.setGrid(theta=theta)

    subhalos = []
    for kw in synthetic_subhalo_specs():
        sh = piemd(co, zl=zl, zs=zs, **kw)
        sh.setGrid(theta=theta)
        subhalos.append(sh)

    return main, subhalos


def build_composite_lens(co, zl, zs, fov_arcsec, npix_model):
    main, subhalos = build_synthetic_lens_components(co, zl, zs, fov_arcsec, npix_model)

    for sh in subhalos:
        main.combinewith(sh)

    main.ka = np.clip(main.ka, 0.0, None)
    return main


def lens_mass_map_msun(co, zl, zs, lens, fov_arcsec, npix_model):
    dl = co.angular_diameter_distance(zl).value
    fov_mpc = np.deg2rad(fov_arcsec / 3600.0) * dl
    pix_mpc = fov_mpc / (npix_model - 1)
    sigma_cr = sigma_crit_msun_mpc2(co, zl, zs)
    return np.clip(lens.ka, 0.0, None) * sigma_cr * pix_mpc**2


def allocate_component_particles(component_masses, nparticles, min_subhalo_particles):
    masses = np.asarray(component_masses, dtype=np.float64)
    counts = np.zeros(masses.size, dtype=np.int64)
    valid = masses > 0.0
    if not np.any(valid):
        return counts

    valid_indices = np.where(valid)[0]
    targets = np.zeros_like(masses)
    targets[valid] = masses[valid] / masses[valid].sum() * float(nparticles)

    if masses.size > 1 and min_subhalo_particles > 0:
        sub_valid = valid.copy()
        sub_valid[0] = False
        n_sub = int(np.sum(sub_valid))
        if n_sub > 0:
            max_sub_total = max(1, int(0.85 * nparticles))
            min_each = min(int(min_subhalo_particles), max(1, max_sub_total // n_sub))
            targets[sub_valid] = np.maximum(targets[sub_valid], float(min_each))
            sub_total = targets[sub_valid].sum()
            if sub_total > max_sub_total:
                targets[sub_valid] *= float(max_sub_total) / sub_total
            targets[0] = max(1.0, float(nparticles) - targets[sub_valid].sum())

    counts[valid] = np.maximum(1, np.floor(targets[valid]).astype(np.int64))
    diff = int(nparticles - counts.sum())
    frac = targets - np.floor(targets)
    if diff > 0:
        order = valid_indices[np.argsort(frac[valid_indices])[::-1]]
        for idx in np.resize(order, diff):
            counts[idx] += 1
    elif diff < 0:
        order = valid_indices[np.argsort(frac[valid_indices])]
        remaining = -diff
        for idx in np.resize(order, remaining * max(1, valid_indices.size)):
            if counts[idx] > 1:
                counts[idx] -= 1
                remaining -= 1
                if remaining == 0:
                    break
    return counts


def sample_particles_from_component_map(
    rng, mass_map, n_particles, component_mass, fov_mpc, pix_mpc, z_sigma_mpc,
    component_id
):
    xpix, ypix = sample2Dimage(mass_map, n=int(n_particles))
    x_mpc = -fov_mpc / 2.0 + ypix * pix_mpc
    y_mpc = -fov_mpc / 2.0 + xpix * pix_mpc
    if z_sigma_mpc > 0.0:
        z_mpc = rng.normal(0.0, z_sigma_mpc, size=int(n_particles))
    else:
        z_mpc = np.zeros(int(n_particles), dtype=np.float64)
    mass = np.full(int(n_particles), float(component_mass) / float(n_particles), dtype=np.float64)
    component_ids = np.full(int(n_particles), int(component_id), dtype=np.int32)
    return np.column_stack((x_mpc, y_mpc, z_mpc)), mass, component_ids


def make_synthetic_cluster_like(
    co,
    zl,
    zs,
    fov_arcsec=220.0,
    npix_model=900,
    nparticles=300000,
    z_sigma_mpc=0.0,
    seed=17,
    sampling="components",
    min_subhalo_particles=2000,
):
    rng = np.random.default_rng(seed)
    main, subhalos = build_synthetic_lens_components(co, zl, zs, fov_arcsec, npix_model)
    lens = build_composite_lens(co, zl, zs, fov_arcsec, npix_model)

    dl = co.angular_diameter_distance(zl).value
    fov_mpc = np.deg2rad(fov_arcsec / 3600.0) * dl
    pix_mpc = fov_mpc / (npix_model - 1)

    if sampling == "composite":
        mass_map = lens_mass_map_msun(co, zl, zs, lens, fov_arcsec, npix_model)
        total_mass = float(mass_map.sum())
        pos, mass, component_ids = sample_particles_from_component_map(
            rng, mass_map, nparticles, total_mass, fov_mpc, pix_mpc, z_sigma_mpc,
            component_id=0,
        )
        component_info = [
            {
                "name": "composite",
                "mass_msun": total_mass,
                "n_particles": int(nparticles),
            }
        ]
    elif sampling == "components":
        components = [main] + subhalos
        names = ["main"] + [f"subhalo_{i:02d}" for i in range(len(subhalos))]
        component_maps = [
            lens_mass_map_msun(co, zl, zs, comp, fov_arcsec, npix_model)
            for comp in components
        ]
        component_masses = np.array([float(mm.sum()) for mm in component_maps], dtype=np.float64)
        counts = allocate_component_particles(component_masses, nparticles, min_subhalo_particles)

        pos_chunks = []
        mass_chunks = []
        component_id_chunks = []
        component_info = []
        for icomp, (name, mass_map, comp_mass, n_comp) in enumerate(
            zip(names, component_maps, component_masses, counts)
        ):
            component_info.append(
                {
                    "component_id": int(icomp),
                    "name": name,
                    "mass_msun": float(comp_mass),
                    "n_particles": int(n_comp),
                }
            )
            if n_comp <= 0 or comp_mass <= 0.0:
                continue
            pos_i, mass_i, component_id_i = sample_particles_from_component_map(
                rng, mass_map, n_comp, comp_mass, fov_mpc, pix_mpc, z_sigma_mpc,
                component_id=icomp,
            )
            pos_chunks.append(pos_i)
            mass_chunks.append(mass_i)
            component_id_chunks.append(component_id_i)

        pos = np.vstack(pos_chunks) if pos_chunks else np.zeros((0, 3), dtype=np.float64)
        mass = np.concatenate(mass_chunks) if mass_chunks else np.zeros(0, dtype=np.float64)
        component_ids = (
            np.concatenate(component_id_chunks)
            if component_id_chunks
            else np.zeros(0, dtype=np.int32)
        )
        total_mass = float(component_masses.sum())
    else:
        raise ValueError("sampling must be either 'composite' or 'components'.")

    cl = cluster.__new__(cluster)
    cl.snapshot = "synthetic"
    cl.parttype = [1]
    cl.verbose = False
    cl.mass = mass
    cl.pos = pos
    cl.ptype = np.ones(mass.size, dtype=np.int32)
    cl.pid = np.arange(mass.size, dtype=np.int64)
    cl.zl = zl
    cl.co = co
    cl.xc = 0.0
    cl.yc = 0.0
    cl.zc = 0.0
    cl.angx = 0.0
    cl.angy = 0.0
    cl.synthetic_component_info = component_info
    cl.synthetic_total_mass = total_mass
    cl.synthetic_sampling = sampling
    cl.synthetic_z_sigma_mpc = float(z_sigma_mpc)
    cl.synthetic_component_id = component_ids

    bounds = dict(
        xmin=-fov_mpc / 2.0,
        xmax=fov_mpc / 2.0,
        ymin=-fov_mpc / 2.0,
        ymax=fov_mpc / 2.0,
    )
    return cl, lens, bounds, fov_arcsec


def project_synthetic_particles(cl, pX, pY, xmin, xmax, ymin, ymax, zmin, zmax):
    pos = np.array(cl.pos, dtype=np.float64, copy=True)
    pos[:, 0] -= cl.xc
    pos[:, 1] -= cl.yc
    pos[:, 2] -= cl.zc

    if pX != 0.0:
        y = pos[:, 1].copy()
        z = pos[:, 2].copy()
        rad = np.deg2rad(pX)
        cosa = np.cos(rad)
        sina = -np.sin(rad)
        pos[:, 1] = y * cosa - z * sina
        pos[:, 2] = y * sina + z * cosa

    if pY != 0.0:
        x = pos[:, 0].copy()
        z = pos[:, 2].copy()
        rad = np.deg2rad(pY)
        cosa = np.cos(rad)
        sina = -np.sin(rad)
        pos[:, 2] = z * cosa - x * sina
        pos[:, 0] = z * sina + x * cosa

    isel = (
        (pos[:, 2] >= zmin)
        & (pos[:, 2] <= zmax)
        & (pos[:, 0] >= xmin)
        & (pos[:, 0] <= xmax)
        & (pos[:, 1] >= ymin)
        & (pos[:, 1] <= ymax)
    )
    component_id = getattr(cl, "synthetic_component_id", np.zeros(cl.mass.size, dtype=np.int32))
    return pos[isel, :], cl.mass[isel], component_id[isel]


def build_synthetic_internal_split_map(
    cl,
    npix,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    pX,
    pY,
    use_numba,
    kernel,
    kernel_gamma,
    main_nb,
    main_min_hsml,
    main_max_hsml,
    subhalo_nb,
    subhalo_min_hsml,
    subhalo_max_hsml,
):
    pos, mass, component_id = project_synthetic_particles(
        cl, pX=pX, pY=pY, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax
    )
    img = np.zeros((npix, npix), dtype=np.float64)
    component_summary = []

    groups = [
        ("main", component_id == 0, main_nb, main_min_hsml, main_max_hsml),
        ("subhalos", component_id > 0, subhalo_nb, subhalo_min_hsml, subhalo_max_hsml),
    ]
    for name, mask, nb, min_hsml, max_hsml in groups:
        n_selected = int(np.sum(mask))
        mass_selected = float(np.sum(mass[mask])) if n_selected else 0.0
        component_summary.append(
            {
                "name": name,
                "n_selected": n_selected,
                "mass_msun": mass_selected,
                "nb": int(nb),
                "min_hsml": float(min_hsml),
                "max_hsml": float(max_hsml),
            }
        )
        if n_selected == 0:
            continue
        img += cluster_module._mass_map_sph_native(
            pos=pos[mask, :],
            mass=mass[mask],
            npix=npix,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            nb=nb,
            min_hsml=min_hsml,
            max_hsml=max_hsml,
            use_numba=use_numba,
            kernel=kernel,
            kernel_gamma=kernel_gamma,
        )

    return img, component_summary


def parse_parttypes(parttypes):
    if isinstance(parttypes, (list, tuple, np.ndarray)):
        return [int(x) for x in parttypes]
    if parttypes is None:
        return [1]
    out = []
    for item in str(parttypes).split(","):
        it = item.strip()
        if it:
            out.append(int(it))
    if len(out) == 0:
        raise ValueError("No valid particle types provided in --parttypes.")
    return out


def load_center_from_subfind(subfind_file, index=0):
    sf = subfind(subfind_file)
    if index < 0 or index >= len(sf.gpos):
        raise IndexError(f"subfind index {index} out of range [0, {len(sf.gpos)-1}]")
    xc, yc, zc = sf.gpos[index]
    return float(xc), float(yc), float(zc)


def generate_maps_snapshot_like_test_massmap(
    snapshot,
    subfind_file,
    subfind_index,
    parttypes,
    fov_arcsec,
    npix_map,
    zmin,
    zmax,
    nb_old,
    nb_tmp,
    min_hsml,
    max_hsml,
    hdf5=False,
    fsample=0.0,
    pX=0.0,
    pY=0.0,
    use_numba=True,
    alignIT=True,
    kernel_internal="cubic_spline",
    kernel_gamma_internal=None,
):
    """
    Build old/new SPH mass maps following the same logic used in test_massmap.py:
    1) read center from subfind
    2) loop over particle types
    3) build one cluster object per particle type
    4) sum the projected maps.
    """
    xc, yc, zc = load_center_from_subfind(subfind_file, index=subfind_index)

    # Build a reference cluster to get cosmology and lens redshift.
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

    mm_kwargs_old = dict(
        pX=pX,
        pY=pY,
        npix=npix_map,
        xmin=bounds["xmin"],
        xmax=bounds["xmax"],
        ymin=bounds["ymin"],
        ymax=bounds["ymax"],
        zmin=zmin,
        zmax=zmax,
        nb=nb_old,
        min_hsml=min_hsml,
        max_hsml=max_hsml,
    )
    mm_kwargs_tmp = dict(mm_kwargs_old)
    mm_kwargs_tmp["nb"] = nb_tmp

    # Same IT used in test_massmap.py
    IT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    img_old = np.zeros((npix_map, npix_map), dtype=np.float64)
    img_tmp = np.zeros((npix_map, npix_map), dtype=np.float64)
    t_old_total = 0.0
    t_tmp_total = 0.0

    per_parttype = []

    for itype in parttypes:
        cl_i = cluster(
            snapshot=snapshot,
            parttype=[itype],
            xc=xc,
            yc=yc,
            zc=zc,
            IT=IT,
            alignIT=alignIT,
            hdf5=hdf5,
            fsample=fsample,
        )

        t0 = time.perf_counter()
        this_old = cl_i.massMapSPH(**mm_kwargs_old)
        dt_old = time.perf_counter() - t0
        t_old_total += dt_old

        # Diagnostics: check whether hsml clipping saturates sensitivity to nb_tmp.
        pos_rel = cl_i.pos.copy()
        pos_rel[:, 0] = pos_rel[:, 0] - cl_i.xc
        pos_rel[:, 1] = pos_rel[:, 1] - cl_i.yc
        pos_rel[:, 2] = pos_rel[:, 2] - cl_i.zc

        if pX != 0.0:
            y = pos_rel[:, 1].copy()
            z = pos_rel[:, 2].copy()
            rad = np.deg2rad(pX)
            cosa = np.cos(rad)
            sina = -np.sin(rad)
            pos_rel[:, 1] = y * cosa - z * sina
            pos_rel[:, 2] = y * sina + z * cosa

        if pY != 0.0:
            x = pos_rel[:, 0].copy()
            z = pos_rel[:, 2].copy()
            rad = np.deg2rad(pY)
            cosa = np.cos(rad)
            sina = -np.sin(rad)
            pos_rel[:, 2] = z * cosa - x * sina
            pos_rel[:, 0] = z * sina + x * cosa

        isel = (
            (pos_rel[:, 2] >= zmin)
            & (pos_rel[:, 2] <= zmax)
            & (pos_rel[:, 0] >= bounds["xmin"])
            & (pos_rel[:, 0] <= bounds["xmax"])
            & (pos_rel[:, 1] >= bounds["ymin"])
            & (pos_rel[:, 1] <= bounds["ymax"])
        )
        pos_sel = pos_rel[isel, :]
        hsml_tmp = cluster_module._compute_hsml_knn(
            pos_sel, nb=int(max(1, nb_tmp)), min_hsml=min_hsml, max_hsml=max_hsml
        )

        t0 = time.perf_counter()
        this_tmp = cl_i.massMapSPH_internal(
            **mm_kwargs_tmp,
            use_numba=use_numba,
            kernel=kernel_internal,
            kernel_gamma=kernel_gamma_internal,
        )
        dt_tmp = time.perf_counter() - t0
        t_tmp_total += dt_tmp

        img_old += this_old
        img_tmp += this_tmp

        per_parttype.append(
            {
                "parttype": int(itype),
                "time_old_sec": float(dt_old),
                "time_tmp_sec": float(dt_tmp),
                "mass_old": float(this_old.sum()),
                "mass_tmp": float(this_tmp.sum()),
                "nb_old": int(nb_old),
                "nb_tmp": int(nb_tmp),
                "kernel_internal": str(kernel_internal),
                "kernel_gamma_internal": None if kernel_gamma_internal is None else float(kernel_gamma_internal),
                "n_selected": int(pos_sel.shape[0]),
                "hsml_tmp_min": float(np.min(hsml_tmp)) if hsml_tmp.size else float("nan"),
                "hsml_tmp_p50": float(np.percentile(hsml_tmp, 50)) if hsml_tmp.size else float("nan"),
                "hsml_tmp_p90": float(np.percentile(hsml_tmp, 90)) if hsml_tmp.size else float("nan"),
                "hsml_tmp_max": float(np.max(hsml_tmp)) if hsml_tmp.size else float("nan"),
                "hsml_tmp_frac_at_min": float(np.mean(np.isclose(hsml_tmp, min_hsml, rtol=0.0, atol=min_hsml * 1e-6))) if hsml_tmp.size else float("nan"),
                "hsml_tmp_frac_at_max": float(np.mean(np.isclose(hsml_tmp, max_hsml, rtol=0.0, atol=max_hsml * 1e-6))) if hsml_tmp.size else float("nan"),
            }
        )

    snapshot_meta = {
        "xc": float(xc),
        "yc": float(yc),
        "zc": float(zc),
        "bounds_mpc": {k: float(v) for k, v in bounds.items()},
        "parttypes": [int(x) for x in parttypes],
        "per_parttype": per_parttype,
        "alignIT": bool(alignIT),
        "IT": IT.tolist(),
    }

    return cl_ref, img_old, img_tmp, t_old_total, t_tmp_total, bounds, snapshot_meta


def build_deflector_from_mass_map(co, zl, zs, mass_map, fov_mass_arcsec, nray, fov_ray_arcsec):
    kwargs = {"zl": zl, "zs": zs, "fov": fov_mass_arcsec, "method": "deflection_fft"}
    rt = raytracer(co, mass_map, Nray=nray, FOVray=fov_ray_arcsec, fromfile=False, **kwargs)
    df = deflector(co, angx=rt.a1, angy=rt.a2, zl=zl, zs=zs)
    theta = np.linspace(-fov_ray_arcsec / 2.0, fov_ray_arcsec / 2.0, nray)
    df.setGrid(theta=theta, compute_potential=False)
    return df


def compute_lensing_metrics(df, ggsl_minsize=0.5, ggsl_maxsize=5.0, ggsl_dmax=200.0):
    metrics = {}

    tl = df.tancl()
    rl = df.radcl()

    metrics["n_tancl"] = int(len(tl))
    metrics["n_radcl"] = int(len(rl))
    metrics["thetaE"] = float(df.thetaE(clt=tl)) if len(tl) > 0 else float("nan")
    metrics["thetaEv"] = [float(x) for x in (df.thetaEv(clt=tl) if len(tl) > 0 else np.array([]))]
    metrics["multImaCrossSection"] = float(df.multImaCrossSection())
    metrics["ggslCrossSection"] = float(
        df.ggslCrossSection(clt=tl, minsize=ggsl_minsize, maxsize=ggsl_maxsize, dmax=ggsl_dmax)
    )

    _, cs_mult = df.imageMultiplicity()
    metrics["imageMultiplicityCrossSections"] = [float(x) for x in cs_mult]

    return metrics, tl, rl


def sample_sources_from_main_caustic(df, nsrc=40, seed=101):
    rng = np.random.default_rng(seed)
    tl = df.tancl()
    if len(tl) == 0:
        return np.array([]), np.array([])

    caut = df.getCaustics(tl)
    if len(caut) == 0:
        return np.array([]), np.array([])

    try:
        sx, sy = df.random_sources_in_caustic(caut[0], buffer_size=0.0, number=max(5 * nsrc, nsrc))
    except Exception:
        return np.array([]), np.array([])

    if len(sx) == 0:
        return np.array([]), np.array([])

    nsel = min(nsrc, len(sx))
    inds = rng.choice(np.arange(len(sx)), size=nsel, replace=False)
    return sx[inds], sy[inds]


def compare_multiple_images(df_old, df_tmp, sx, sy, zs):
    if len(sx) == 0:
        return {
            "nsources": 0,
            "n_multi_old": 0,
            "n_multi_tmp": 0,
            "mean_nimg_old": float("nan"),
            "mean_nimg_tmp": float("nan"),
            "mean_abs_delta_nimg": float("nan"),
        }

    nimg_old = []
    nimg_tmp = []

    for xsrc, ysrc in zip(sx, sy):
        ps_old = pointsrc(gl=df_old, ys1=float(xsrc), ys2=float(ysrc), zs=zs, flux=1.0, Npix=64, refine=False)
        ps_tmp = pointsrc(gl=df_tmp, ys1=float(xsrc), ys2=float(ysrc), zs=zs, flux=1.0, Npix=64, refine=False)
        nimg_old.append(len(ps_old.xi1))
        nimg_tmp.append(len(ps_tmp.xi1))

    nimg_old = np.array(nimg_old)
    nimg_tmp = np.array(nimg_tmp)

    return {
        "nsources": int(len(sx)),
        "n_multi_old": int(np.sum(nimg_old > 1)),
        "n_multi_tmp": int(np.sum(nimg_tmp > 1)),
        "mean_nimg_old": float(np.mean(nimg_old)),
        "mean_nimg_tmp": float(np.mean(nimg_tmp)),
        "mean_abs_delta_nimg": float(np.mean(np.abs(nimg_old - nimg_tmp))),
    }


def plot_critical_lines(ax, df, cl_list, color="white", ls="-", lw=1.0, alpha=0.9, label=None):
    for i, c in enumerate(cl_list):
        x, y = df.getCritPoints(c, pixel_units=False)
        ax.plot(x, y, ls=ls, color=color, lw=lw, alpha=alpha, label=label if i == 0 else None)


def build_synthetic_reference_lens(co, zl, zs, fov_ray_arcsec, nray):
    lens = build_composite_lens(co, zl, zs, fov_ray_arcsec, nray)
    return lens

def residual_region_metrics(img_old, img_tmp, border_width_pix):
    res = img_tmp - img_old
    eps = 1e-30
    ny, nx = img_old.shape
    bw = int(max(1, min(border_width_pix, nx // 2 - 1, ny // 2 - 1)))

    border = np.zeros_like(img_old, dtype=bool)
    border[:bw, :] = True
    border[-bw:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True
    center = ~border

    def _metrics(mask):
        old = np.abs(img_old[mask]).sum()
        l1 = np.abs(res[mask]).sum() / max(old, eps)
        rms = np.sqrt(np.mean(res[mask] ** 2))
        return float(l1), float(rms), int(mask.sum())

    l1_b, rms_b, npix_b = _metrics(border)
    l1_c, rms_c, npix_c = _metrics(center)
    return {
        "border_width_pix": int(bw),
        "border": {"l1_relative": l1_b, "rms_absolute": rms_b, "npix": npix_b},
        "center": {"l1_relative": l1_c, "rms_absolute": rms_c, "npix": npix_c},
    }


def main():
    parser = argparse.ArgumentParser(description="Compare massMapSPH and massMapSPH_internal.")
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "synthetic", "snapshot"],
        help="Input mode. 'auto' uses snapshot mode if --snapshot is provided.",
    )
    parser.add_argument("--snapshot", type=str, default=None, help="Path to simulation snapshot.")
    parser.add_argument("--parttypes", type=str, default="1", help="Comma-separated particle types (e.g. '0,1,4').")
    parser.add_argument("--hdf5", action="store_true", help="Read snapshot as HDF5.")
    parser.add_argument("--fsample", type=float, default=0.0, help="Particle sampling factor/fraction passed to cluster.")
    parser.add_argument("--subfind-file", type=str, default=None, help="Subfind file used to retrieve halo center.")
    parser.add_argument("--subfind-index", type=int, default=0, help="Index of halo center in subfind catalog.")
    parser.add_argument("--nparticles", type=int, default=250000)
    parser.add_argument("--npix-model", type=int, default=850)
    parser.add_argument("--npix-map", type=int, default=1024)
    parser.add_argument("--nray", type=int, default=1024)
    parser.add_argument("--fov-arcsec", type=float, default=220.0)
    parser.add_argument("--fov-ray-arcsec", type=float, default=180.0)
    parser.add_argument(
        "--z-sigma-mpc",
        type=float,
        default=0.0,
        help=(
            "Synthetic line-of-sight particle scatter in Mpc. The default is a "
            "thin lens plane; nonzero values can inflate 3D kNN smoothing lengths."
        ),
    )
    parser.add_argument(
        "--synthetic-sampling",
        type=str,
        default="components",
        choices=["components", "composite"],
        help=(
            "Synthetic particle sampling mode. 'components' samples the main halo "
            "and subhalos separately so compact subhalos are not under-sampled."
        ),
    )
    parser.add_argument(
        "--min-subhalo-particles",
        type=int,
        default=2000,
        help=(
            "Minimum target particle count for each nonzero synthetic subhalo when "
            "--synthetic-sampling components is used."
        ),
    )
    parser.add_argument("--zmax-mpc", type=float, default=0.25)
    parser.add_argument("--nb", type=int, default=32, help="Global nb fallback if --nb-old/--nb-internal are not set.")
    parser.add_argument("--nb-old", type=int, default=None, help="Neighbor count for massMapSPH (old).")
    parser.add_argument("--nb-internal", type=int, default=None, help="Neighbor count for massMapSPH_internal.")
    parser.add_argument("--nb-tmp", type=int, default=None, help="Deprecated alias for --nb-internal.")
    parser.add_argument(
        "--synthetic-internal-split-components",
        action="store_true",
        help=(
            "In synthetic mode, build the internal map as main-halo plus subhalo "
            "SPH maps with separate smoothing settings."
        ),
    )
    parser.add_argument("--main-nb-internal", type=int, default=None)
    parser.add_argument("--main-min-hsml", type=float, default=None)
    parser.add_argument("--main-max-hsml", type=float, default=None)
    parser.add_argument("--subhalo-nb-internal", type=int, default=None)
    parser.add_argument("--subhalo-min-hsml", type=float, default=None)
    parser.add_argument("--subhalo-max-hsml", type=float, default=None)
    parser.add_argument(
        "--kernel-internal",
        type=str,
        default="cubic_spline",
        choices=["cubic_spline", "wendland_c2"],
        help="SPH kernel used by massMapSPH_internal.",
    )
    parser.add_argument(
        "--kernel-gamma-internal",
        type=float,
        default=None,
        help=(
            "Compact-support radius in units of hsml for the internal Wendland-C2 "
            "kernel. Defaults to the SWIFT/SWIFTSIMIO value 1.936492."
        ),
    )
    parser.add_argument("--min-hsml", type=float, default=1e-4)
    parser.add_argument("--max-hsml", type=float, default=0.05)
    parser.add_argument("--zl", type=float, default=0.4)
    parser.add_argument("--zs", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--nsrc", type=int, default=40)
    parser.add_argument("--prefix", type=str, default="massMapSPH_compare")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--pX", type=float, default=0.0, help="Rotation around X (deg).")
    parser.add_argument("--pY", type=float, default=0.0, help="Rotation around Y (deg).")
    args = parser.parse_args()

    co = FlatLambdaCDM(H0=70.0, Om0=0.3)
    if args.mode == "auto":
        run_mode = "snapshot" if args.snapshot is not None else "synthetic"
    else:
        run_mode = args.mode
    if args.nb_internal is not None and args.nb_tmp is not None:
        raise ValueError("Use only one between --nb-internal and --nb-tmp.")
    nb_old = int(args.nb if args.nb_old is None else args.nb_old)
    nb_tmp_cli = args.nb_internal if args.nb_internal is not None else args.nb_tmp
    nb_tmp = int(args.nb if nb_tmp_cli is None else nb_tmp_cli)
    main_nb_internal = int(nb_old if args.main_nb_internal is None else args.main_nb_internal)
    main_min_hsml = float(args.min_hsml if args.main_min_hsml is None else args.main_min_hsml)
    main_max_hsml = float(args.max_hsml if args.main_max_hsml is None else args.main_max_hsml)
    subhalo_nb_internal = int(nb_tmp if args.subhalo_nb_internal is None else args.subhalo_nb_internal)
    subhalo_min_hsml = float(args.min_hsml if args.subhalo_min_hsml is None else args.subhalo_min_hsml)
    subhalo_max_hsml = float(args.max_hsml if args.subhalo_max_hsml is None else args.subhalo_max_hsml)
    kernel_id_internal = cluster_module._sph_kernel_id(args.kernel_internal)
    kernel_gamma_internal = cluster_module._sph_kernel_gamma(
        kernel_id_internal, args.kernel_gamma_internal
    )

    if run_mode == "snapshot":
        print ("Running in snapshot mode...")
        print ("working with snapshot:", args.snapshot)
        print ("the corresponding subfind file is:", args.subfind_file)
        if args.snapshot is None:
            raise ValueError("Snapshot mode requires --snapshot.")
        if args.subfind_file is None:
            raise ValueError("Snapshot mode requires --subfind-file (as in test_massmap.py).")
        parttypes = parse_parttypes(args.parttypes)
        cl_ref, img_old, img_tmp, t_old, t_tmp, bounds, snapshot_meta = generate_maps_snapshot_like_test_massmap(
            snapshot=args.snapshot,
            subfind_file=args.subfind_file,
            subfind_index=args.subfind_index,
            parttypes=parttypes,
            fov_arcsec=args.fov_arcsec,
            npix_map=args.npix_map,
            zmin=-args.zmax_mpc,
            zmax=args.zmax_mpc,
            nb_old=nb_old,
            nb_tmp=nb_tmp,
            min_hsml=args.min_hsml,
            max_hsml=args.max_hsml,
            hdf5=args.hdf5,
            fsample=args.fsample,
            pX=args.pX,
            pY=args.pY,
            use_numba=(not args.no_numba),
            alignIT=True,
            kernel_internal=args.kernel_internal,
            kernel_gamma_internal=args.kernel_gamma_internal,
        )
        fov_arcsec = args.fov_arcsec
        zl_eff = cl_ref.zl
        co_eff = cl_ref.co
    else:
        print ("Running in synthetic mode...")
        cl, _lens_ref, bounds, fov_arcsec = make_synthetic_cluster_like(
            co=co,
            zl=args.zl,
            zs=args.zs,
            fov_arcsec=args.fov_arcsec,
            npix_model=args.npix_model,
            nparticles=args.nparticles,
            z_sigma_mpc=args.z_sigma_mpc,
            seed=args.seed,
            sampling=args.synthetic_sampling,
            min_subhalo_particles=args.min_subhalo_particles,
        )
        zl_eff = args.zl
        co_eff = co

    if run_mode != "snapshot":
        mm_kwargs = dict(
            pX=args.pX,
            pY=args.pY,
            npix=args.npix_map,
            xmin=bounds["xmin"],
            xmax=bounds["xmax"],
            ymin=bounds["ymin"],
            ymax=bounds["ymax"],
            zmin=-args.zmax_mpc,
            zmax=args.zmax_mpc,
            nb=nb_old,
            min_hsml=args.min_hsml,
            max_hsml=args.max_hsml,
        )

        mm_kwargs_tmp = dict(
            pX=args.pX,
            pY=args.pY,
            npix=args.npix_map,
            xmin=bounds["xmin"],
            xmax=bounds["xmax"],
            ymin=bounds["ymin"],
            ymax=bounds["ymax"],
            zmin=-args.zmax_mpc,
            zmax=args.zmax_mpc,
            nb=nb_tmp,
            min_hsml=args.min_hsml,
            max_hsml=args.max_hsml,
        )

        t0 = time.perf_counter()
        img_old = cl.massMapSPH(**mm_kwargs)
        t_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        synthetic_split_summary = None
        if args.synthetic_internal_split_components:
            img_tmp, synthetic_split_summary = build_synthetic_internal_split_map(
                cl=cl,
                npix=args.npix_map,
                xmin=bounds["xmin"],
                xmax=bounds["xmax"],
                ymin=bounds["ymin"],
                ymax=bounds["ymax"],
                zmin=-args.zmax_mpc,
                zmax=args.zmax_mpc,
                pX=args.pX,
                pY=args.pY,
                use_numba=(not args.no_numba),
                kernel=args.kernel_internal,
                kernel_gamma=args.kernel_gamma_internal,
                main_nb=main_nb_internal,
                main_min_hsml=main_min_hsml,
                main_max_hsml=main_max_hsml,
                subhalo_nb=subhalo_nb_internal,
                subhalo_min_hsml=subhalo_min_hsml,
                subhalo_max_hsml=subhalo_max_hsml,
            )
        else:
            img_tmp = cl.massMapSPH_internal(
                **mm_kwargs_tmp,
                use_numba=(not args.no_numba),
                kernel=args.kernel_internal,
                kernel_gamma=args.kernel_gamma_internal,
            )
        t_tmp = time.perf_counter() - t0
    else:
        synthetic_split_summary = None

    res = img_tmp - img_old
    abs_old_sum = float(np.sum(np.abs(img_old)))
    l1_rel = float(np.sum(np.abs(res)) / abs_old_sum) if abs_old_sum > 0 else float("nan")
    rms_abs = float(np.sqrt(np.mean(res**2)))
    corr = float(np.corrcoef(img_old.ravel(), img_tmp.ravel())[0, 1])
    pixel_mpc = (bounds["xmax"] - bounds["xmin"]) / float(args.npix_map)
    border_width_pix = int(np.ceil((kernel_gamma_internal * args.max_hsml) / max(pixel_mpc, 1e-30)))
    region_metrics = residual_region_metrics(img_old, img_tmp, border_width_pix=border_width_pix)

    df_old = build_deflector_from_mass_map(
        co_eff, zl_eff, args.zs, img_old, fov_arcsec, args.nray, args.fov_ray_arcsec
    )
    df_tmp = build_deflector_from_mass_map(
        co_eff, zl_eff, args.zs, img_tmp, fov_arcsec, args.nray, args.fov_ray_arcsec
    )

    metrics_old, tl_old, rl_old = compute_lensing_metrics(df_old)
    metrics_tmp, tl_tmp, rl_tmp = compute_lensing_metrics(df_tmp)
    df_ref = None
    metrics_ref = None
    tl_ref = np.array([])
    rl_ref = np.array([])
    if run_mode == "synthetic":
        df_ref = build_synthetic_reference_lens(co_eff, zl_eff, args.zs, args.fov_ray_arcsec, args.nray)
        metrics_ref, tl_ref, rl_ref = compute_lensing_metrics(df_ref)

    sx, sy = sample_sources_from_main_caustic(df_old, nsrc=args.nsrc, seed=args.seed + 11)
    src_stats = compare_multiple_images(df_old, df_tmp, sx, sy, args.zs)

    summary = {
        "runtime_sec": {
            "massMapSPH": float(t_old),
            "massMapSPH_internal": float(t_tmp),
            "speedup_internal_vs_old": float(t_old / t_tmp) if t_tmp > 0 else float("nan"),
        },
        "mass_conservation": {
            "sum_old": float(img_old.sum()),
            "sum_tmp": float(img_tmp.sum()),
            "relative_delta": float((img_tmp.sum() - img_old.sum()) / img_old.sum()) if img_old.sum() != 0 else float("nan"),
        },
        "map_residuals": {
            "l1_relative": l1_rel,
            "rms_absolute": rms_abs,
            "pearson_r": corr,
        },
        "map_residuals_by_region": region_metrics,
        "lensing_old": metrics_old,
        "lensing_tmp": metrics_tmp,
        "point_source_multiplicity": src_stats,
        "data_mode": run_mode,
        "nb_effective": {"old": int(nb_old), "internal": int(nb_tmp)},
        "kernel_effective": {
            "old": "sphviewer_or_internal_default",
            "internal": args.kernel_internal,
            "internal_gamma": float(kernel_gamma_internal),
        },
        "backend_selection": {
            "sphviewer_available": bool(getattr(cluster_module, "_SPHVIEWER_AVAILABLE", False)),
            "massMapSPH_backend": getattr(cluster_module, "_SPHVIEWER_BACKEND", "internal"),
        },
        "config": vars(args),
    }

    if run_mode == "synthetic":
        summary["lensing_analytic_reference"] = metrics_ref
        summary["synthetic_sampling"] = {
            "mode": getattr(cl, "synthetic_sampling", args.synthetic_sampling),
            "z_sigma_mpc": float(getattr(cl, "synthetic_z_sigma_mpc", args.z_sigma_mpc)),
            "total_mass_msun": float(getattr(cl, "synthetic_total_mass", img_old.sum())),
            "components": getattr(cl, "synthetic_component_info", []),
        }
        if synthetic_split_summary is not None:
            summary["synthetic_internal_split_components"] = synthetic_split_summary

    if run_mode == "snapshot":
        summary["snapshot_info"] = {
            "snapshot": args.snapshot,
            "parttypes": parttypes,
            "zl_from_snapshot": float(zl_eff),
            "center_used": [snapshot_meta["xc"], snapshot_meta["yc"], snapshot_meta["zc"]],
            "subfind_file": args.subfind_file,
            "subfind_index": int(args.subfind_index),
            "hdf5": bool(args.hdf5),
            "fsample": float(args.fsample),
            "alignIT": bool(snapshot_meta["alignIT"]),
            "IT": snapshot_meta["IT"],
            "parttype_timings_and_masses": snapshot_meta["per_parttype"],
            "bounds_mpc": snapshot_meta["bounds_mpc"],
        }

    out_json = Path(f"{args.prefix}_summary.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(2, 3, figsize=(18, 11))

    vmax = max(np.percentile(img_old, 99.7), np.percentile(img_tmp, 99.7))
    eps = max(1e-20, 1e-8 * vmax)

    im0 = ax[0, 0].imshow(np.log10(img_old + eps), origin="lower", cmap="magma")
    ax[0, 0].set_title("log10 massMapSPH")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(np.log10(img_tmp + eps), origin="lower", cmap="magma")
    ax[0, 1].set_title(f"log10 massMapSPH_internal ({args.kernel_internal})")
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    frac = res / np.maximum(np.abs(img_old), eps)
    im2 = ax[0, 2].imshow(frac, origin="lower", cmap="coolwarm", vmin=-0.5, vmax=0.5)
    ax[0, 2].set_title("(tmp-old)/old")
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

    ext = [-args.fov_ray_arcsec / 2.0, args.fov_ray_arcsec / 2.0, -args.fov_ray_arcsec / 2.0, args.fov_ray_arcsec / 2.0]

    ax[1, 0].imshow(df_old.ka, origin="lower", extent=ext, cmap="viridis")
    plot_critical_lines(ax[1, 0], df_old, tl_old, color="white", ls="-", lw=1.2, label="old tan")
    plot_critical_lines(ax[1, 0], df_tmp, tl_tmp, color="cyan", ls="--", lw=1.0, label="internal tan")
    if run_mode == "synthetic":
        plot_critical_lines(ax[1, 0], df_ref, tl_ref, color="gold", ls="-.", lw=1.2, label="analytic tan")
        plot_critical_lines(ax[1, 0], df_old, rl_old, color="white", ls=":", lw=0.9, alpha=0.75, label="old rad")
        plot_critical_lines(ax[1, 0], df_tmp, rl_tmp, color="cyan", ls=":", lw=0.9, alpha=0.75, label="internal rad")
        plot_critical_lines(ax[1, 0], df_ref, rl_ref, color="gold", ls=":", lw=1.0, alpha=0.85, label="analytic rad")
        ax[1, 0].legend(loc="upper right", fontsize=7, frameon=True)
    ax[1, 0].set_title(f"critical lines on old kappa; internal={args.kernel_internal}")

    ax[1, 1].imshow(df_tmp.ka, origin="lower", extent=ext, cmap="viridis")
    plot_critical_lines(ax[1, 1], df_tmp, tl_tmp, color="white", ls="-", lw=1.2, label="internal tan")
    plot_critical_lines(ax[1, 1], df_old, tl_old, color="magenta", ls="--", lw=1.0, label="old tan")
    if run_mode == "synthetic":
        plot_critical_lines(ax[1, 1], df_ref, tl_ref, color="gold", ls="-.", lw=1.2, label="analytic tan")
        plot_critical_lines(ax[1, 1], df_tmp, rl_tmp, color="white", ls=":", lw=0.9, alpha=0.75, label="internal rad")
        plot_critical_lines(ax[1, 1], df_old, rl_old, color="magenta", ls=":", lw=0.9, alpha=0.75, label="old rad")
        plot_critical_lines(ax[1, 1], df_ref, rl_ref, color="gold", ls=":", lw=1.0, alpha=0.85, label="analytic rad")
        ax[1, 1].legend(loc="upper right", fontsize=7, frameon=True)
    ax[1, 1].set_title(f"critical lines on internal kappa ({args.kernel_internal})")

    labels = [
        "thetaE",
        "MI cs",
        "GGSL cs",
        "N tanCL",
        "N radCL",
    ]
    vals_old = [
        metrics_old["thetaE"],
        metrics_old["multImaCrossSection"],
        metrics_old["ggslCrossSection"],
        metrics_old["n_tancl"],
        metrics_old["n_radcl"],
    ]
    vals_tmp = [
        metrics_tmp["thetaE"],
        metrics_tmp["multImaCrossSection"],
        metrics_tmp["ggslCrossSection"],
        metrics_tmp["n_tancl"],
        metrics_tmp["n_radcl"],
    ]

    x = np.arange(len(labels))
    if run_mode == "synthetic":
        vals_ref = [
            metrics_ref["thetaE"],
            metrics_ref["multImaCrossSection"],
            metrics_ref["ggslCrossSection"],
            metrics_ref["n_tancl"],
            metrics_ref["n_radcl"],
        ]
        w = 0.26
        ax[1, 2].bar(x - w, vals_old, w, label="old")
        ax[1, 2].bar(x, vals_tmp, w, label="internal")
        ax[1, 2].bar(x + w, vals_ref, w, label="analytic")
    else:
        w = 0.38
        ax[1, 2].bar(x - w / 2.0, vals_old, w, label="old")
        ax[1, 2].bar(x + w / 2.0, vals_tmp, w, label="internal")
    ax[1, 2].set_xticks(x)
    ax[1, 2].set_xticklabels(labels, rotation=20)
    ax[1, 2].legend(loc="best")
    ax[1, 2].set_title("Lensing diagnostics")

    fig.suptitle(
        f"old={t_old:.2f}s internal={t_tmp:.2f}s speedup={summary['runtime_sec']['speedup_internal_vs_old']:.2f}x | "
        f"L1rel={l1_rel:.3e} corr={corr:.6f}"
    )
    fig.tight_layout()

    out_png = Path(f"{args.prefix}_comparison.png")
    fig.savefig(out_png, dpi=180)

    print("Saved summary:", out_json)
    print("Saved figure:", out_png)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
