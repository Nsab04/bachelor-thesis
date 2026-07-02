"""
Mass-matching test for one lens galaxy:

1) compute PIEMD total mass from LensTool parameters,
2) build deprojected Sersic 3D density from catalog (Re, n, mu),
3) compute k such that M_PIEMD = k * M_Sersic,3D.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.special import gamma

from pyLensLib.lenstool import (
    findInBlock,
    getRADECfromXY,
    getRef_RA_DEC,
    getXYfromPotentiel,
    readLenstoolBlock,
    selectPotentielByType,
)


G_KPC = 4.302e-6  # kpc (km/s)^2 / Msun
DEFAULT_PAR = "/Users/maxmen3/stiva/pietro_models/M0416_B22.par"
DEFAULT_CATALOG = "/Users/maxmen3/projects/Pierpaoli/m0416.dat"
DEFAULT_README = "/Users/maxmen3/projects/Pierpaoli/ReadMe.txt"


def _parse_vizier_bytes_table(readme_text: str, data_basename: str):
    sec_pat = re.compile(r"Byte-by-byte Description of file:\s*(.*)")
    lines = readme_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        m = sec_pat.search(line)
        if m and data_basename in m.group(1):
            start_idx = i
            break
    if start_idx is None:
        for i, line in enumerate(lines):
            if sec_pat.search(line):
                start_idx = i
                break
    if start_idx is None:
        raise ValueError("No 'Byte-by-byte' section found in ReadMe.")

    row_pat = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+[A-Za-z0-9.]+\s+\S+\s+([A-Za-z0-9_]+)")
    colspecs, names = [], []
    for line in lines[start_idx + 1 :]:
        if line.strip().startswith("Byte-by-byte Description of file:"):
            break
        m = row_pat.match(line)
        if m:
            a, b, label = m.groups()
            colspecs.append((int(a) - 1, int(b)))
            names.append(label)
        elif colspecs and not line.strip():
            break
    if not colspecs:
        raise ValueError("Could not parse column specs from ReadMe.")
    return colspecs, names


def load_cluster(readme_path, data_path):
    with open(readme_path, encoding="utf-8", errors="replace") as f:
        readme_text = f.read()
    colspecs, names = _parse_vizier_bytes_table(readme_text, os.path.basename(data_path))
    return pd.read_fwf(data_path, colspecs=colspecs, names=names)


def pick_sersic_columns(df_cat, band):
    re_col = f"Re{band}"
    n_col = f"N{band}"
    mu_col = f"mu{band}"
    required = [re_col, n_col, mu_col]
    missing = [c for c in required if c not in df_cat.columns]
    if missing:
        raise KeyError(f"Missing required columns for band={band}: {missing}")
    return {"re": re_col, "n": n_col, "mu": mu_col}


def piemd_mtotal(r_cut, sigma_v):
    return np.pi * sigma_v**2 / G_KPC * r_cut


def b_n_sersic(n):
    n = float(n)
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)


def p_n_prugniel_simien(n):
    n = float(n)
    return 1.0 - 0.6097 / n + 0.05463 / (n**2)


def sersic_rho_3d(r, re_kpc, n, mu_e):
    r = np.asarray(r, dtype=float)
    re_kpc = float(re_kpc)
    n = float(n)
    mu_e = float(mu_e)
    rr = np.maximum(r, np.finfo(float).eps) / re_kpc
    bn = b_n_sersic(n)
    pn = p_n_prugniel_simien(n)
    rho_e = 10.0 ** (-0.4 * mu_e)
    return rho_e * np.power(rr, -pn) * np.exp(-bn * (np.power(rr, 1.0 / n) - 1.0))


def sersic_mtotal_3d_analytic(re_kpc, n, mu_e):
    re_kpc = float(re_kpc)
    n = float(n)
    mu_e = float(mu_e)
    bn = b_n_sersic(n)
    pn = p_n_prugniel_simien(n)
    alpha = n * (3.0 - pn)
    rho_e = 10.0 ** (-0.4 * mu_e)
    return 4.0 * np.pi * rho_e * re_kpc**3 * n * np.exp(bn) * (bn ** (-alpha)) * gamma(alpha)


def sersic_mtotal_3d_numeric(re_kpc, n, mu_e, rmin_factor=1.0e-6, rmax_factor=1.0e4, npts=20000):
    rmin = max(rmin_factor * re_kpc, 1.0e-6)
    rmax = max(rmax_factor * re_kpc, 1.0e2)
    r = np.logspace(np.log10(rmin), np.log10(rmax), int(npts))
    rho = sersic_rho_3d(r, re_kpc, n, mu_e)
    return np.trapezoid(4.0 * np.pi * r**2 * rho, r)


def main():
    parser = argparse.ArgumentParser(
        description="Compute k from mass matching: M_PIEMD = k * M_Sersic,3D for one matched lens galaxy."
    )
    parser.add_argument("--parfile", default=DEFAULT_PAR, help="LensTool parameter file.")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help="Photo-morphology catalog file.")
    parser.add_argument("--explain", default=DEFAULT_README, help="VizieR ReadMe for fixed-width parsing.")
    parser.add_argument("--band", default="F160W", help="Band token for Sersic parameters (e.g. F160W).")
    parser.add_argument("--lens-index", type=int, default=1, help="Lens index in LensTool potentiel list.")
    parser.add_argument("--tolerance-arcsec", type=float, default=0.1, help="Sky-match tolerance in arcsec.")
    args = parser.parse_args()

    potentiel_all = readLenstoolBlock(best_par=args.parfile, block_name="potentiel")
    potentiel = selectPotentielByType(potentiel_all, ptype="gal")
    df_cat = load_cluster(readme_path=args.explain, data_path=args.catalog)
    scol = pick_sersic_columns(df_cat, args.band)

    x, y = getXYfromPotentiel(potentiel)
    ra_ref, dec_ref = getRef_RA_DEC(args.parfile)
    ra_lens, dec_lens = getRADECfromXY(x, y, ra_ref, dec_ref)

    coords_lens = SkyCoord(ra=ra_lens * u.deg, dec=dec_lens * u.deg)
    coords_cat = SkyCoord(ra=df_cat["RAdeg"].values * u.deg, dec=df_cat["DEdeg"].values * u.deg)
    idx_cat, d2d, _ = coords_lens.match_to_catalog_sky(coords_cat)
    matched = d2d < (args.tolerance_arcsec * u.arcsec)
    lens_idx = np.where(matched)[0]
    cat_idx = idx_cat[matched]
    match_map = dict(zip(lens_idx, cat_idx))

    j = int(args.lens_index)
    if j not in match_map:
        raise RuntimeError(f"lens_index={j} has no catalog match within {args.tolerance_arcsec} arcsec.")
    j_cat = int(match_map[j])
    crow = df_cat.iloc[j_cat]

    re_kpc = float(crow[scol["re"]])
    n_ser = float(crow[scol["n"]])
    mu_e = float(crow[scol["mu"]])
    sigma0 = float(findInBlock(potentiel[j], "v_disp")) * np.sqrt(3.0 / 2.0)
    r_core = float(findInBlock(potentiel[j], "core_radius_kpc"))
    r_cut = float(findInBlock(potentiel[j], "cut_radius_kpc"))

    m_piemd = piemd_mtotal(r_cut=r_cut, sigma_v=sigma0)
    m_ser_ana = sersic_mtotal_3d_analytic(re_kpc=re_kpc, n=n_ser, mu_e=mu_e)
    m_ser_num = sersic_mtotal_3d_numeric(re_kpc=re_kpc, n=n_ser, mu_e=mu_e)

    k_ana = m_piemd / m_ser_ana
    k_num = m_piemd / m_ser_num

    print(f"lens_index={j}, cat_index={j_cat}, band={args.band}")
    print(f"PIEMD params: sigma0={sigma0:.6g} km/s, r_core={r_core:.6g} kpc, r_cut={r_cut:.6g} kpc")
    print(f"Sersic params: Re={re_kpc:.6g} kpc, n={n_ser:.6g}, mu_e={mu_e:.6g}")
    print(f"M_PIEMD_total = {m_piemd:.12e} Msun")
    print(f"M_Sersic3D_shape (analytic) = {m_ser_ana:.12e} [arb]")
    print(f"M_Sersic3D_shape (numeric)  = {m_ser_num:.12e} [arb]")
    print(f"k (analytic mass match) = {k_ana:.12e}")
    print(f"k (numeric  mass match) = {k_num:.12e}")
    print(f"relative difference k_num vs k_ana = {(k_num - k_ana) / k_ana:.6e}")


if __name__ == "__main__":
    main()
