#!/usr/bin/env python3
"""
LTsimcat: simulate Lenstool-like multiple-image catalogs from deflection maps.

Main workflow:
1) Read deflection-angle maps from FITS:
   - single file with two HDUs (a1 in primary, a2 in extension), or
   - two separate files (a1 and a2).
2) Build a pyLensLib deflector with cosmology and lens/source redshifts.
3) Generate a source catalog (srccatalog) from kwargs in YAML.
4) Build source planes, select sources inside caustics, and ray-trace point sources.
5) Save multiple images in Lenstool-like catalog format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
import yaml
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors

# Prefer local repository sources over stale site-packages installs when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def read_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_fits_image(path: str, hdu_index: int = 0) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = np.asarray(hdul[hdu_index].data, dtype=float)
        header = hdul[hdu_index].header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image in {path}[{hdu_index}], got shape {data.shape}")
    return data, header


def load_deflection_maps(cfg: Dict) -> Tuple[np.ndarray, np.ndarray, fits.Header]:
    """
    Load deflection maps from:
    - single FITS file with a1/a2 in two HDUs, or
    - two separate FITS files.
    """
    dcfg = cfg.get("deflection", {})
    single = dcfg.get("single_file")
    x_file = dcfg.get("x_file")
    y_file = dcfg.get("y_file")

    if single:
        hdu_a1 = int(dcfg.get("hdu_a1", 0))
        hdu_a2 = int(dcfg.get("hdu_a2", 1))
        a1, h1 = _read_fits_image(single, hdu_a1)
        a2, h2 = _read_fits_image(single, hdu_a2)
        if a1.shape != a2.shape:
            raise ValueError(f"a1 and a2 shapes differ in single file: {a1.shape} vs {a2.shape}")
        header = h1 if len(h1) >= len(h2) else h2
        return a1, a2, header

    if x_file and y_file:
        hdu_x = int(dcfg.get("hdu_x", 0))
        hdu_y = int(dcfg.get("hdu_y", 0))
        a1, h1 = _read_fits_image(x_file, hdu_x)
        a2, h2 = _read_fits_image(y_file, hdu_y)
        if a1.shape != a2.shape:
            raise ValueError(f"a1 and a2 shapes differ in split files: {a1.shape} vs {a2.shape}")
        header = h1 if len(h1) >= len(h2) else h2
        return a1, a2, header

    raise ValueError(
        "Invalid deflection configuration. Provide either:\n"
        "  deflection.single_file\n"
        "or both:\n"
        "  deflection.x_file and deflection.y_file"
    )


def apply_cli_deflection_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    cfg = dict(cfg) if cfg is not None else {}
    dcfg = dict(cfg.get("deflection", {}))

    has_single = args.angles_file is not None
    has_split_any = (args.angx_file is not None) or (args.angy_file is not None)

    if has_single and has_split_any:
        raise ValueError("Use either --angles-file or --angx-file/--angy-file, not both.")
    if has_split_any and not (args.angx_file and args.angy_file):
        raise ValueError("When using split files, provide both --angx-file and --angy-file.")

    if has_single:
        dcfg["single_file"] = args.angles_file
        dcfg["hdu_a1"] = int(args.hdu_a1)
        dcfg["hdu_a2"] = int(args.hdu_a2)
        dcfg.pop("x_file", None)
        dcfg.pop("y_file", None)
    elif args.angx_file and args.angy_file:
        dcfg["x_file"] = args.angx_file
        dcfg["y_file"] = args.angy_file
        dcfg["hdu_x"] = int(args.hdu_x)
        dcfg["hdu_y"] = int(args.hdu_y)
        dcfg.pop("single_file", None)

    cfg["deflection"] = dcfg
    return cfg


def infer_axes_arcsec(header: fits.Header, shape: Tuple[int, int], cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer coordinate axes (arcsec) from header.

    Priority:
    1) Explicit XMIN/XMAX/YMIN/YMAX
    2) WCS linear terms CD/CDELT + CRPIX
    3) config.pixel_scale_arcsec fallback

    Notes:
    - By default, if XMIN/XMAX/YMIN/YMAX are present, they are interpreted as
      pixel-center bounds and converted to symmetric endpoint axes.
      Set deflection.use_header_bounds_exact=true to use them verbatim.
    - By default, when using WCS linear terms, we construct symmetric endpoint
      axes from pixel scale and map size (FOV = Npix * pixel_scale). This
      matches the pyLensLib convention used in many scripts (e.g. -FOV/2..FOV/2
      inclusive) and avoids half-pixel asymmetries with even-sized grids.
    - Set deflection.wcs_use_pixel_centers=true to use strict pixel-center WCS
      coordinates instead.
    """
    ny, nx = shape
    if all(k in header for k in ("XMIN", "XMAX", "YMIN", "YMAX")):
        dcfg = cfg.get("deflection", {})
        xmin = float(header["XMIN"])
        xmax = float(header["XMAX"])
        ymin = float(header["YMIN"])
        ymax = float(header["YMAX"])

        if bool(dcfg.get("use_header_bounds_exact", False)):
            thetax = np.linspace(xmin, xmax, nx)
            thetay = np.linspace(ymin, ymax, ny)
            return thetax, thetay

        # Default: convert center-bounds to symmetric endpoint-bounds.
        # If bounds are already symmetric endpoints, the correction is negligible.
        pixx = abs((xmax - xmin) / max(nx - 1, 1))
        pixy = abs((ymax - ymin) / max(ny - 1, 1))
        xcen = 0.5 * (xmin + xmax)
        ycen = 0.5 * (ymin + ymax)
        if "CRVAL1" in header:
            xcen = float(header["CRVAL1"]) * 3600.0
        elif (xmin < 0.0 < xmax) and (abs(xcen) <= max(pixx, 1e-12)):
            # Common lens-map convention: header center intended at 0 arcsec.
            xcen = 0.0
        if "CRVAL2" in header:
            ycen = float(header["CRVAL2"]) * 3600.0
        elif (ymin < 0.0 < ymax) and (abs(ycen) <= max(pixy, 1e-12)):
            ycen = 0.0
        fovx = pixx * float(nx)
        fovy = pixy * float(ny)
        thetax = np.linspace(xcen - 0.5 * fovx, xcen + 0.5 * fovx, nx)
        thetay = np.linspace(ycen - 0.5 * fovy, ycen + 0.5 * fovy, ny)
        return thetax, thetay

    cd11 = header.get("CD1_1", header.get("CDELT1"))
    cd22 = header.get("CD2_2", header.get("CDELT2"))
    crpix1 = float(header.get("CRPIX1", (nx + 1) / 2.0))
    crpix2 = float(header.get("CRPIX2", (ny + 1) / 2.0))

    if cd11 is not None and cd22 is not None:
        dcfg = cfg.get("deflection", {})
        if bool(dcfg.get("wcs_use_pixel_centers", False)):
            # Strict pixel-center WCS coordinates.
            crval1 = float(header.get("CRVAL1", 0.0))
            crval2 = float(header.get("CRVAL2", 0.0))
            thetax = crval1 * 3600.0 + (np.arange(nx, dtype=float) + 1.0 - crpix1) * float(cd11) * 3600.0
            thetay = crval2 * 3600.0 + (np.arange(ny, dtype=float) + 1.0 - crpix2) * float(cd22) * 3600.0
            return thetax, thetay

        # Default: symmetric endpoint grid from pixel scale and Npix.
        pixx = abs(float(cd11)) * 3600.0
        pixy = abs(float(cd22)) * 3600.0
        xcen = float(header.get("CRVAL1", 0.0)) * 3600.0
        ycen = float(header.get("CRVAL2", 0.0)) * 3600.0
        fovx = pixx * float(nx)
        fovy = pixy * float(ny)
        thetax = np.linspace(xcen - 0.5 * fovx, xcen + 0.5 * fovx, nx)
        thetay = np.linspace(ycen - 0.5 * fovy, ycen + 0.5 * fovy, ny)
        return thetax, thetay

    pix = cfg.get("deflection", {}).get("pixel_scale_arcsec")
    if pix is None:
        raise ValueError(
            "Cannot infer map axes from header. Missing (XMIN/XMAX/YMIN/YMAX) and (CD/CDELT). "
            "Set deflection.pixel_scale_arcsec in YAML."
        )
    pix = float(pix)
    fovx = float(nx) * pix
    fovy = float(ny) * pix
    thetax = np.linspace(-0.5 * fovx, 0.5 * fovx, nx)
    thetay = np.linspace(-0.5 * fovy, 0.5 * fovy, ny)
    return thetax, thetay


def enforce_increasing_axes(
    a1: np.ndarray,
    a2: np.ndarray,
    thetax: np.ndarray,
    thetay: np.ndarray,
    reorder_maps: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure axis vectors are increasing.

    By default we do NOT reorder `a1/a2` when axis vectors are reversed. This matches
    the internal pyLensLib convention used to generate many deflection maps (array frame),
    where forcing map flips from WCS handedness can produce inconsistent lensing results.

    Set `reorder_maps=True` only for inputs where deflection maps are explicitly stored
    in the same handedness as the WCS axis direction.
    """
    if thetax.size < 2 or thetay.size < 2:
        raise ValueError("Need at least 2 pixels per axis to enforce axis ordering.")

    if thetax[1] < thetax[0]:
        thetax = thetax[::-1]
        if reorder_maps:
            a1 = a1[:, ::-1]
            a2 = a2[:, ::-1]
    if thetay[1] < thetay[0]:
        thetay = thetay[::-1]
        if reorder_maps:
            a1 = a1[::-1, :]
            a2 = a2[::-1, :]
    return a1, a2, thetax, thetay


def build_cosmology(cfg: Dict, header: fits.Header) -> FlatLambdaCDM:
    ccfg = cfg.get("cosmology", {})
    h0 = ccfg.get("H0")
    om0 = ccfg.get("Om0")
    if h0 is None:
        if "H0" in header:
            h0 = float(header["H0"])
        elif "H" in header:
            h0 = float(header["H"]) * 100.0
        else:
            h0 = 70.0
    if om0 is None:
        if "OMEGAM" in header:
            om0 = float(header["OMEGAM"])
        elif "OMEGA" in header:
            om0 = float(header["OMEGA"])
        else:
            om0 = 0.3
    return FlatLambdaCDM(H0=float(h0), Om0=float(om0))


def build_deflector(
    a1: np.ndarray, a2: np.ndarray, header: fits.Header, cfg: Dict
) -> Tuple["deflector", float, float, np.ndarray, np.ndarray]:
    from pyLensLib.deflector import deflector

    lcfg = cfg.get("lens", {})
    # Priority requested by user: if FITS header has ZL/ZS, use those.
    if "ZL" in header:
        zl = float(header["ZL"])
    elif "ZLENS" in header:
        zl = float(header["ZLENS"])
    else:
        zl = float(lcfg.get("zl", 0.4))

    if "ZS" in header:
        zs_ref = float(header["ZS"])
    else:
        zs_ref = float(lcfg.get("zs_ref", 2.0))

    co = build_cosmology(cfg, header)

    thetax, thetay = infer_axes_arcsec(header, a1.shape, cfg)
    reorder_maps = bool(cfg.get("deflection", {}).get("reorder_maps_with_axes", False))
    a1, a2, thetax, thetay = enforce_increasing_axes(
        a1, a2, thetax, thetay, reorder_maps=reorder_maps
    )

    df = deflector(co, angx=a1, angy=a2, zl=zl, zs=zs_ref)
    df.setGrid(thetax=thetax, thetay=thetay)
    return df, zl, zs_ref, thetax, thetay


def prepare_sources(df: "deflector", zl: float, cfg: Dict) -> pd.DataFrame:
    from pyLensLib.srccatalog import srccatalog

    scfg = cfg.get("source_catalog", {})
    kwargs_src = dict(scfg.get("kwargs_src", {}))
    if "FOV" not in kwargs_src:
        kwargs_src["FOV"] = float(max(df.size1, df.size2))
    if "seed" not in kwargs_src and "seed" in scfg:
        kwargs_src["seed"] = int(scfg["seed"])

    cat = srccatalog(**kwargs_src)
    df_src = cat.get_dataframe().copy()
    df_src = df_src[df_src["zgal"] >= zl].copy()
    return df_src


def dlensing_ratio(cosmo: FlatLambdaCDM, z_lens: float, z_source: np.ndarray) -> np.ndarray:
    z_source = np.asarray(z_source, dtype=float)
    d_l = cosmo.angular_diameter_distance(z_lens).value
    d_s = cosmo.angular_diameter_distance(z_source).value
    d_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = d_l * d_ls / d_s
    return ratio


def generate_source_planes(cosmo: FlatLambdaCDM, z_lens: float, z_source_max: float, n_planes: int) -> np.ndarray:
    dl_max = float(dlensing_ratio(cosmo, z_lens, np.array([z_source_max]))[0])
    zs_arr = np.linspace(z_lens, z_source_max, 1000)
    dl_arr = dlensing_ratio(cosmo, z_lens, zs_arr)
    dl_equi = np.linspace(0.0, dl_max, int(n_planes))
    z_planes = np.interp(dl_equi, dl_arr, zs_arr, left=np.nan, right=np.nan)
    z_planes[0] = 0.0
    return z_planes


def assign_sources_to_planes(
    df_src: pd.DataFrame, z_lens: float, cfg: Dict, cosmo: FlatLambdaCDM
) -> pd.DataFrame:
    """
    Assign each source to the closest source plane in lensing-distance space.
    """
    if len(df_src) == 0:
        out = df_src.copy()
        out.loc[:, "plane_index"] = np.array([], dtype=int)
        out.loc[:, "plane_z"] = np.array([], dtype=float)
        return out

    pcfg = cfg.get("source_planes", {})
    n_planes = int(pcfg.get("n_planes", 100))
    z_source_max = float(pcfg.get("z_source_max", 11.0))
    z_planes = generate_source_planes(cosmo, z_lens=z_lens, z_source_max=z_source_max, n_planes=n_planes)

    z_source = df_src["zgal"].values.astype(float)
    plane_indices = np.zeros(len(z_source), dtype=int)
    behind = z_source > float(z_lens)
    if np.any(behind):
        z_diff = np.abs(z_source[behind, np.newaxis] - z_planes[np.newaxis, :])
        plane_indices[behind] = np.argmin(z_diff, axis=1)

    out = df_src.copy()
    out.loc[:, "plane_index"] = plane_indices
    out.loc[:, "plane_z"] = z_planes[plane_indices]
    return out


def select_sources_inside_caustics(
    df: "deflector", df_src: pd.DataFrame, cfg: Dict
) -> pd.DataFrame:
    if len(df_src) == 0:
        return df_src.copy()

    pcfg = cfg.get("source_planes", {})
    caustic_buffer = float(pcfg.get("caustic_buffer_arcsec", 0.0))

    df_src = df_src.copy()
    if "plane_z" not in df_src.columns:
        raise ValueError("select_sources_inside_caustics requires a 'plane_z' column. Call assign_sources_to_planes first.")

    selected = []
    for z_plane in np.unique(df_src["plane_z"]):
        if float(z_plane) <= float(df.zl):
            continue
        sel = df_src["plane_z"] == z_plane
        if sel.sum() == 0:
            continue
        df.change_redshift(float(z_plane))
        _ = df.multImaCrossSection(buffer_size=caustic_buffer)

        xs = df_src.loc[sel, "x"].values
        ys = df_src.loc[sel, "y"].values
        xs_pix, ys_pix = df.arcsec2pixel(xs, ys)
        inside = df.points_in_UU(xs_pix, ys_pix)
        picked = df_src.loc[sel].copy()
        picked = picked[inside]
        selected.append(picked)
        #print (f"Plane z={z_plane:.3f}: {sel.sum()} sources, {inside.sum()} inside caustics")

    if len(selected) == 0:
        return df_src.iloc[0:0].copy()
    out = pd.concat(selected, ignore_index=True)
    return out


def find_multiple_images(df: "deflector", df_sel: pd.DataFrame, zs_ref: float, cfg: Dict) -> List[Dict]:
    from pyLensLib.pointsrc import pointsrc

    images: List[Dict] = []
    if len(df_sel) == 0:
        return images

    pcfg = cfg.get("pointsrc", {})
    use_lenstronomy = bool(pcfg.get("use_lenstronomy", False))
    refine = bool(pcfg.get("refine", True))
    img_maglim = float(cfg.get("image_catalog", {}).get("img_maglim", 99.0))
    fov = float(max(df.size1, df.size2))
    npix = int(df.a1.shape[0])

    df_sel = df_sel.reset_index(drop=True)
    if "plane_z" in df_sel.columns:
        zvals = df_sel["plane_z"].values
    else:
        zvals = df_sel["zgal"].values
    mags = df_sel["mag"].values
    xs = df_sel["x"].values
    ys = df_sel["y"].values

    unique_z = np.unique(zvals)
    for z_now in unique_z:
        df.change_redshift(float(z_now))
        idxs = np.where(zvals == z_now)[0]
        for i_src in idxs:
            ps = pointsrc(
                size=fov,
                Npix=npix,
                gl=df,
                use_lenstronomy=use_lenstronomy,
                refine=refine,
                ys1=float(xs[i_src]),
                ys2=float(ys[i_src]),
                flux=1.0,
                zs=float(z_now),
            )
            xi, yi, mui = ps.find_images()
            if len(xi) <= 1:
                continue

            mui = np.asarray(mui, dtype=float)
            img_mag = -2.5 * np.log10(np.maximum(np.abs(mui), 1e-30)) + float(mags[i_src])
            keep = img_mag < img_maglim
            if np.sum(keep) <= 1:
                continue

            xi = np.asarray(xi, dtype=float)[keep]
            yi = np.asarray(yi, dtype=float)[keep]
            img_mag = np.asarray(img_mag, dtype=float)[keep]

            order = np.argsort(img_mag)
            xi = xi[order]
            yi = yi[order]
            img_mag = img_mag[order]

            src_id = int(i_src + 1)
            for j in range(len(xi)):
                images.append(
                    {
                        "ID": f"{src_id}.{j + 1}",
                        "RA": float(xi[j]),
                        "DEC": float(yi[j]),
                        "a": 1.0,
                        "b": 1.0,
                        "theta": 0.0,
                        "z": float(z_now),
                        "mag": float(img_mag[j]),
                    }
                )

    df.change_redshift(zs_ref)
    return images


def write_lenstool_like_catalog(path: str, images: List[Dict], ref_ra: float = 0.0, ref_dec: float = 0.0) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"#REFERENCE 3 {ref_ra} {ref_dec}\n")
        for e in images:
            row = [e["ID"], e["RA"], e["DEC"], e["a"], e["b"], e["theta"], e["z"], e["mag"]]
            f.write(" ".join(str(x) for x in row) + "\n")


def write_sources_catalog(path: str, df_sel: pd.DataFrame, ref_ra: float = 0.0, ref_dec: float = 0.0) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"#REFERENCE 3 {ref_ra} {ref_dec}\n")
        for i, row in df_sel.reset_index(drop=True).iterrows():
            rid = str(i + 1)
            vals = [rid, float(row["x"]), float(row["y"]), 1.0, 1.0, float(row["PA"]), float(row["zgal"]), float(row["mag"])]
            f.write(" ".join(str(x) for x in vals) + "\n")


def make_validation_figure(
    df: "deflector",
    images: List[Dict],
    thetax: np.ndarray,
    thetay: np.ndarray,
    zs_planes: List[float],
    out_path: str,
    show_convergence: bool = True,
) -> None:
    """
    Validation plot: critical lines at selected source-plane redshifts + multiple images.
    Multiple images are color-coded by source-plane redshift; critical lines use same z color coding.
    Images are labeled by ID (e.g. 1.1, 1.2, ...).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)

    if show_convergence:
        kappa = np.asarray(df.ka, dtype=float)
        vmax = np.nanpercentile(kappa[np.isfinite(kappa)], 99.0) if np.any(np.isfinite(kappa)) else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        ax.imshow(
            kappa,
            origin="lower",
            extent=[thetax.min(), thetax.max(), thetay.min(), thetay.max()],
            cmap="gray_r",
            alpha=0.35,
            vmin=0.0,
            vmax=vmax,
        )

    zvals = [float(z) for z in zs_planes if float(z) > float(df.zl)]
    zimg = np.array([float(e["z"]) for e in images], dtype=float) if len(images) > 0 else np.array([], dtype=float)
    z_candidates = zvals + zimg[np.isfinite(zimg)].tolist()
    if len(z_candidates) > 0:
        zmin = float(np.min(z_candidates))
        zmax = float(np.max(z_candidates))
    else:
        zmin = max(float(df.zl) + 0.1, 1.0)
        zmax = zmin + 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0
    norm_z = mcolors.Normalize(vmin=zmin, vmax=zmax)
    cmap_z = cm.get_cmap("viridis")

    z_before = float(df.zs)
    for zc in zvals:
        df.change_redshift(float(zc))
        col = cmap_z(norm_z(zc))
        # Tangential and radial critical lines, same color for this source redshift.
        for cl in df.tancl():
            x, y = df.getCritPoints(cl)
            ax.plot(x, y, "-", color=col, lw=1.6, alpha=0.95)
        for cl in df.radcl():
            x, y = df.getCritPoints(cl)
            ax.plot(x, y, "--", color=col, lw=1.2, alpha=0.95)
    df.change_redshift(z_before)

    if len(images) > 0:
        ximg = np.array([float(e["RA"]) for e in images], dtype=float)
        yimg = np.array([float(e["DEC"]) for e in images], dtype=float)
        # Keep image color coding by source redshift in the same range used for critical lines.
        if np.any(np.isfinite(zimg)):
            zmin_i = min(zmin, float(np.nanmin(zimg)))
            zmax_i = max(zmax, float(np.nanmax(zimg)))
            norm_img = mcolors.Normalize(vmin=zmin_i, vmax=zmax_i if zmax_i > zmin_i else zmin_i + 1.0)
        else:
            norm_img = norm_z
        sc = ax.scatter(
            ximg,
            yimg,
            c=zimg,
            cmap=cmap_z,
            norm=norm_img,
            s=26,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("Source-plane redshift")
        # Label each image with its catalog ID (e.g., 1.1, 1.2, ...)
        for e in images:
            x = float(e["RA"])
            y = float(e["DEC"])
            lab = str(e["ID"])
            ax.text(
                x + 0.35,
                y + 0.35,
                lab,
                fontsize=6,
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.2),
            )

    ax.set_xlim(float(thetax.min()), float(thetax.max()))
    ax.set_ylim(float(thetay.min()), float(thetay.max()))
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")
    ax.set_aspect("equal")
    if len(zvals) > 0:
        ztxt = ",".join(f"{z:.3g}" for z in zvals)
        ax.set_title(f"Validation: critical lines (z={ztxt}) and multiple images")
    else:
        ax.set_title("Validation: no multiply-imaged source planes found")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_caustics_validation_figure(
    df: "deflector",
    df_sel: pd.DataFrame,
    thetax: np.ndarray,
    thetay: np.ndarray,
    zs_planes: List[float],
    out_path: str,
) -> None:
    """
    Validation plot: caustics at selected source-plane redshifts + selected sources.
    Source markers are color-coded by assigned source plane redshift (plane_z), using
    the same colormap/norm adopted for the caustics.
    Sources are labeled by integer IDs (1, 2, 3, ...).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)

    zvals = [float(z) for z in zs_planes if float(z) > float(df.zl)]
    if len(df_sel) > 0 and "plane_z" in df_sel.columns:
        zsrc = df_sel["plane_z"].to_numpy(dtype=float)
        zsrc = zsrc[np.isfinite(zsrc)]
        zsrc = zsrc[zsrc > float(df.zl)]
    else:
        zsrc = np.array([], dtype=float)

    z_candidates = zvals + zsrc.tolist()
    zmin = min(z_candidates) if len(z_candidates) > 0 else max(float(df.zl) + 0.1, 1.0)
    zmax = max(z_candidates) if len(z_candidates) > 0 else 6.0
    if zmax <= zmin:
        zmax = zmin + 1.0
    norm_z = mcolors.Normalize(vmin=zmin, vmax=zmax)
    cmap_z = cm.get_cmap("viridis")

    z_before = float(df.zs)
    for zc in zvals:
        df.change_redshift(float(zc))
        col = cmap_z(norm_z(zc))
        for cau in df.getCaustics(df.tancl()):
            x, y = df.getCausticPoints(cau)
            ax.plot(x, y, "-", color=col, lw=1.6, alpha=0.95)
        for cau in df.getCaustics(df.radcl()):
            x, y = df.getCausticPoints(cau)
            ax.plot(x, y, "--", color=col, lw=1.2, alpha=0.95)
    df.change_redshift(z_before)

    if len(df_sel) > 0:
        xs = df_sel["x"].to_numpy(dtype=float)
        ys = df_sel["y"].to_numpy(dtype=float)
        if "plane_z" in df_sel.columns:
            zs = df_sel["plane_z"].to_numpy(dtype=float)
        else:
            zs = df_sel["zgal"].to_numpy(dtype=float)
        sc = ax.scatter(
            xs,
            ys,
            c=zs,
            cmap=cmap_z,
            norm=norm_z,
            s=20,
            edgecolors="white",
            linewidths=0.30,
            alpha=0.90,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("Source-plane redshift")
        # Label sources by their source ID used in the catalogs (1,2,3,...)
        for i, row in df_sel.reset_index(drop=True).iterrows():
            ax.text(
                float(row["x"]) + 0.35,
                float(row["y"]) + 0.35,
                str(i + 1),
                fontsize=6,
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.2),
            )

    ax.set_xlim(float(thetax.min()), float(thetax.max()))
    ax.set_ylim(float(thetay.min()), float(thetay.max()))
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")
    ax.set_aspect("equal")
    if len(zvals) > 0:
        ztxt = ",".join(f"{z:.3g}" for z in zvals)
        ax.set_title(f"Validation: caustics (z={ztxt}) and selected sources")
    else:
        ax.set_title("Validation: no multiply-imaged source planes found")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_redshift_histogram_figure(
    df_src_all: pd.DataFrame,
    df_sel: pd.DataFrame,
    images: List[Dict],
    out_path: str,
    bins: int = 30,
) -> None:
    """
    Validation histograms of source redshifts:
    - panel 1: zgal distributions
    - panel 2: zplane distributions

    Each panel shows all generated sources, selected sources, and multiply-imaged sources.
    """
    z_all_gal = df_src_all["zgal"].to_numpy(dtype=float) if len(df_src_all) > 0 else np.array([], dtype=float)
    z_sel_gal = df_sel["zgal"].to_numpy(dtype=float) if len(df_sel) > 0 else np.array([], dtype=float)
    if len(df_src_all) > 0 and "plane_z" in df_src_all.columns:
        z_all_plane = df_src_all["plane_z"].to_numpy(dtype=float)
    else:
        z_all_plane = z_all_gal.copy()
    if len(df_sel) > 0 and "plane_z" in df_sel.columns:
        z_sel_plane = df_sel["plane_z"].to_numpy(dtype=float)
    else:
        z_sel_plane = z_sel_gal.copy()

    # Reconstruct unique source IDs in df_sel that generated multiple images.
    z_mult_gal = np.array([], dtype=float)
    z_mult_plane = np.array([], dtype=float)
    if len(images) > 0 and len(df_sel) > 0:
        src_ids = []
        for e in images:
            sid = str(e.get("ID", "")).split(".")[0]
            if sid.isdigit():
                src_ids.append(int(sid))
        if len(src_ids) > 0:
            src_ids = np.unique(np.asarray(src_ids, dtype=int))
            idx0 = src_ids - 1
            valid = (idx0 >= 0) & (idx0 < len(df_sel))
            if np.any(valid):
                rows = df_sel.iloc[idx0[valid]]
                z_mult_gal = rows["zgal"].to_numpy(dtype=float)
                if "plane_z" in rows.columns:
                    z_mult_plane = rows["plane_z"].to_numpy(dtype=float)
                else:
                    z_mult_plane = z_mult_gal.copy()

    z_combined = np.concatenate(
        [z_all_gal, z_sel_gal, z_mult_gal, z_all_plane, z_sel_plane, z_mult_plane]
    ) if (len(z_all_gal) + len(z_sel_gal) + len(z_mult_gal) + len(z_all_plane) + len(z_sel_plane) + len(z_mult_plane)) > 0 else np.array([0.0, 1.0])
    zmin = float(np.nanmin(z_combined))
    zmax = float(np.nanmax(z_combined))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        zmin, zmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)

    ax = axes[0]
    ax.hist(z_all_gal, bins=bins, range=(zmin, zmax), histtype="step", lw=1.8, color="#1f77b4", label=f"All (N={len(z_all_gal)})")
    ax.hist(z_sel_gal, bins=bins, range=(zmin, zmax), histtype="step", lw=1.8, color="#ff7f0e", label=f"Inside caustics (N={len(z_sel_gal)})")
    ax.hist(z_mult_gal, bins=bins, range=(zmin, zmax), histtype="step", lw=1.8, color="#2ca02c", label=f"Multiply imaged (N={len(z_mult_gal)})")
    ax.set_xlabel("zgal")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.set_title("Validation: zgal distributions")
    ax.legend(loc="best", frameon=False)

    ax = axes[1]
    ax.hist(z_all_plane, bins=bins, range=(zmin, zmax), histtype="step", lw=1.8, color="#1f77b4", label=f"All (N={len(z_all_plane)})")
    ax.hist(z_sel_plane, bins=bins, range=(zmin, zmax), histtype="step", lw=1.8, color="#ff7f0e", label=f"Inside caustics (N={len(z_sel_plane)})")
    ax.hist(z_mult_plane, bins=bins, range=(zmin, zmax), histtype="step", lw=1.8, color="#2ca02c", label=f"Multiply imaged (N={len(z_mult_plane)})")
    ax.set_xlabel("zplane")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.set_title("Validation: zplane distributions")
    ax.legend(loc="best", frameon=False)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lenstool-like source/image catalogs from deflection maps.")
    parser.add_argument("config", help="YAML configuration file")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional prefix for output files. If set, output names become "
             "<prefix>_sources.cat, <prefix>_images.cat, and <prefix>_val_*.png.",
    )
    parser.add_argument("--angles-file", default=None,
                        help="Single FITS file containing both deflection components (a1/a2 in two HDUs).")
    parser.add_argument("--hdu-a1", type=int, default=0,
                        help="HDU index for a1 in --angles-file (default: 0).")
    parser.add_argument("--hdu-a2", type=int, default=1,
                        help="HDU index for a2 in --angles-file (default: 1).")
    parser.add_argument("--angx-file", default=None,
                        help="FITS file containing x-component deflection map (a1).")
    parser.add_argument("--angy-file", default=None,
                        help="FITS file containing y-component deflection map (a2).")
    parser.add_argument("--hdu-x", type=int, default=0,
                        help="HDU index for a1 in --angx-file (default: 0).")
    parser.add_argument("--hdu-y", type=int, default=1,
                        help="HDU index for a2 in --angy-file (default: 1).")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    cfg = apply_cli_deflection_overrides(cfg, args)
    outdir = Path(cfg.get("output", {}).get("output_dir", "ltsimcat_output"))
    outdir.mkdir(parents=True, exist_ok=True)

    a1, a2, header = load_deflection_maps(cfg)
    df, zl, zs_ref, thetax, thetay = build_deflector(a1, a2, header, cfg)

    print(f"Loaded deflection maps: shape={a1.shape}")
    print(f"Lens redshift zl={zl:.4f}, reference source redshift zs_ref={zs_ref:.4f}")
    print(f"Grid: x=[{thetax.min():.3f},{thetax.max():.3f}] arcsec, y=[{thetay.min():.3f},{thetay.max():.3f}] arcsec")

    df_src = prepare_sources(df, zl, cfg)
    print(f"Generated sources behind lens: {len(df_src)}")

    df_src_all = assign_sources_to_planes(df_src, zl, cfg, df.co)
    df_sel = select_sources_inside_caustics(df, df_src_all, cfg)
    print(f"Sources inside caustics: {len(df_sel)}")

    images = find_multiple_images(df, df_sel, zs_ref, cfg)
    print(f"Total multiple images saved: {len(images)}")

    ocfg = cfg.get("output", {})
    if args.prefix:
        src_file = outdir / f"{args.prefix}_sources.cat"
        img_file = outdir / f"{args.prefix}_images.cat"
    else:
        src_file = outdir / ocfg.get("source_catalog", "simulated_sources.cat")
        img_file = outdir / ocfg.get("image_catalog", "simulated_images.cat")
    write_sources_catalog(src_file.as_posix(), df_sel, ref_ra=0.0, ref_dec=0.0)
    write_lenstool_like_catalog(img_file.as_posix(), images, ref_ra=0.0, ref_dec=0.0)

    print(f"Wrote source catalog: {src_file}")
    print(f"Wrote image catalog:  {img_file}")

    vcfg = cfg.get("validation", {})
    if bool(vcfg.get("enabled", False)):
        # Use only source-plane redshifts where at least one source is actually multiply imaged.
        zs_planes = sorted({float(e["z"]) for e in images if float(e["z"]) > float(zl)})
        if args.prefix:
            out_fig = outdir / f"{args.prefix}_val_critical_lines_images.png"
            out_fig_cau = outdir / f"{args.prefix}_val_caustics_sources.png"
            out_fig_zh = outdir / f"{args.prefix}_val_source_redshift_hist.png"
        else:
            out_fig = outdir / vcfg.get("output_figure", "validation_critical_lines_images.png")
            out_fig_cau = outdir / vcfg.get("output_caustics_figure", "validation_caustics_sources.png")
            out_fig_zh = outdir / vcfg.get("output_redshift_histogram", "validation_source_redshift_hist.png")
        show_conv = bool(vcfg.get("show_convergence", True))
        bins_zh = int(vcfg.get("redshift_hist_bins", 30))
        make_validation_figure(
            df=df,
            images=images,
            thetax=thetax,
            thetay=thetay,
            zs_planes=zs_planes,
            out_path=out_fig.as_posix(),
            show_convergence=show_conv,
        )
        make_caustics_validation_figure(
            df=df,
            df_sel=df_sel,
            thetax=thetax,
            thetay=thetay,
            zs_planes=zs_planes,
            out_path=out_fig_cau.as_posix(),
        )
        make_redshift_histogram_figure(
            df_src_all=df_src_all,
            df_sel=df_sel,
            images=images,
            out_path=out_fig_zh.as_posix(),
            bins=bins_zh,
        )
        print(f"Wrote validation figure: {out_fig}")
        print(f"Wrote caustics figure:   {out_fig_cau}")
        print(f"Wrote z-hist figure:     {out_fig_zh}")


if __name__ == "__main__":
    main()
