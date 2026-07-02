"""
Identify galaxies with problematic tNFW residual fits from fitGalaxies_mcmc_h5 outputs.

Problematic means that rho_DM = rho_PIEMD - rho_Jaffe is either:
1) negative in the central region (r <= 1 kpc), or
2) rising toward larger radii in the center (positive log-slope in 0.02-0.2 kpc).

Inputs:
  - lens_galaxy_fits.csv
  - lens_galaxy_fits.h5

Outputs:
  - problematic_tnfw_all_galaxies.csv
  - problematic_tnfw_galaxies.csv
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


G_KPC_KMS2_MSUN = 4.302e-6  # kpc (km/s)^2 / Msun


def jaffe_rho(r, m_star, re_kpc):
    """3D Jaffe density profile in Msun/kpc^3."""
    rj = re_kpc
    return (m_star * rj) / (4.0 * np.pi * r**2 * (r + rj) ** 2)


def piemd_rho(r, r_core, r_cut, sigma_v):
    """3D PIEMD density profile in Msun/kpc^3."""
    return (
        (sigma_v**2)
        / (2.0 * np.pi * G_KPC_KMS2_MSUN)
        * (r_cut + r_core)
        / (r_core**2 * r_cut)
        / (1.0 + r**2 / r_core**2)
        / (1.0 + r**2 / r_cut**2)
    )


def central_problem_flags(r, rho_dm):
    """
    Return (negative_center, rising_center, r_first_positive_kpc).

    Definitions:
    - negative_center: any rho_dm <= 0 for r <= 1 kpc
    - rising_center: positive slope dlog(rho_dm)/dlog(r) over 0.02-0.2 kpc
      (computed only from points with rho_dm > 0; requires >= 8 points)
    """
    center_mask = r <= 1.0
    slope_mask = (r >= 0.02) & (r <= 0.2)

    negative_center = bool(np.any(rho_dm[center_mask] <= 0.0))

    rr = r[slope_mask]
    yy = rho_dm[slope_mask]
    pos = yy > 0.0
    rising_center = False
    slope_center = np.nan
    if np.count_nonzero(pos) >= 8:
        x = np.log10(rr[pos])
        y = np.log10(yy[pos])
        slope_center = float(np.polyfit(x, y, 1)[0])
        rising_center = slope_center > 0.0

    pos_idx = np.where(rho_dm > 0.0)[0]
    r_first_positive = float(r[pos_idx[0]]) if pos_idx.size else np.nan

    return negative_center, rising_center, slope_center, r_first_positive


def main():
    base = Path(__file__).resolve().parent
    csv_path = base / "lens_galaxy_fits.csv"
    h5_path = base / "lens_galaxy_fits.h5"

    if not csv_path.exists() or not h5_path.exists():
        raise FileNotFoundError(
            "Missing input files. Expected both lens_galaxy_fits.csv and lens_galaxy_fits.h5 in Test/."
        )

    df = pd.read_csv(csv_path)
    r = np.logspace(-2, 3, 1000)  # same radial range used in fitGalaxies_mcmc_h5.py

    with h5py.File(h5_path, "r") as f:
        piemd_chain = f["piemd_params"][()]  # shape (Ngal, Nsamp, 2): [v_disp, cut_kpc]
        jaffe_params = f["jaffe_params"][()]  # shape (Ngal, 3): [M_star, Re_kpc, r_J_kpc]
        lens_index_h5 = f["lens_index"][()]

    if not np.array_equal(lens_index_h5, df["lens_index"].to_numpy()):
        raise RuntimeError("lens_index ordering mismatch between CSV and HDF5.")

    records = []
    for i, row in df.iterrows():
        lens_index = int(row["lens_index"])
        cat_index = int(row["cat_index"])
        m_star = float(row["M_star"])
        re = float(row["Re_F160W"])
        r_core = float(row["core_radius_kpc"])
        sigma = float(row["v_disp"])
        r_cut = float(row["cut_radius_kpc"])

        rho_j = jaffe_rho(r, m_star, re)
        rho_p = piemd_rho(r, r_core, r_cut, sigma)
        rho_dm = rho_p - rho_j

        neg_center, rise_center, slope_center, r_first_pos = central_problem_flags(r, rho_dm)

        # MCMC sample-level problematic fraction using per-sample PIEMD parameters.
        nsamp = piemd_chain.shape[1]
        bad_count = 0
        for s in range(nsamp):
            sigma_s = float(piemd_chain[i, s, 0])
            r_cut_s = float(piemd_chain[i, s, 1])
            rho_dm_s = piemd_rho(r, r_core, r_cut_s, sigma_s) - rho_j
            neg_s, rise_s, _, _ = central_problem_flags(r, rho_dm_s)
            if neg_s or rise_s:
                bad_count += 1

        records.append(
            {
                "lens_index": lens_index,
                "cat_index": cat_index,
                "F160W": float(row["F160W"]),
                "fstar_Re": float(row["fstar_Re"]),
                "stellar_mass_capped": bool(row["stellar_mass_capped"]),
                "negative_center": neg_center,
                "rising_center": rise_center,
                "problematic": bool(neg_center or rise_center),
                "slope_center_logrho_logr": slope_center,
                "r_first_positive_kpc": r_first_pos,
                "problematic_sample_fraction": bad_count / float(nsamp),
            }
        )

    out = pd.DataFrame(records).sort_values("lens_index").reset_index(drop=True)
    out_problem = out[out["problematic"]].copy().sort_values("lens_index").reset_index(drop=True)

    all_out = base / "problematic_tnfw_all_galaxies.csv"
    prob_out = base / "problematic_tnfw_galaxies.csv"
    out.to_csv(all_out, index=False)
    out_problem.to_csv(prob_out, index=False)

    print(f"Total galaxies analyzed: {len(out)}")
    print(f"Problematic galaxies: {len(out_problem)}")
    print(f"Saved: {all_out}")
    print(f"Saved: {prob_out}")
    print("Problematic lens_index list:")
    print(", ".join(str(x) for x in out_problem["lens_index"].tolist()))


if __name__ == "__main__":
    main()
