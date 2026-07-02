"""
Standalone validation script for the Elliptical Power-Law (EPL) lens model.

Run from the project root (or any folder):
    python Test/test_epl.py

Optional:
    python Test/test_epl.py --show
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from pyLensLib.epl import epl
from pyLensLib.sie import sie


def finite_diff_potential_gradient(model, x, y, h=1.0e-5):
    """Numerical gradient of the potential at (x, y)."""
    dphi_dx = (model.potential(x + h, y) - model.potential(x - h, y)) / (2.0 * h)
    dphi_dy = (model.potential(x, y + h) - model.potential(x, y - h)) / (2.0 * h)
    return dphi_dx, dphi_dy


def check_gradient_consistency(co):
    """Check alpha = grad(phi) away from the lens center."""
    model = epl(
        co,
        zl=0.5,
        zs=2.0,
        theta_E=1.3,
        q=0.75,
        pa=0.35,
        t=1.2,
        theta_c=1.0e-4,
    )

    rng = np.random.default_rng(12345)
    pts = rng.uniform(-2.0, 2.0, size=(80, 2))
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[r > 0.35]

    rel_err = []
    for x, y in pts:
        a1, a2 = model.angle(np.array([x]), np.array([y]))
        fd1, fd2 = finite_diff_potential_gradient(model, float(x), float(y))

        s1 = max(1.0, abs(float(a1[0])))
        s2 = max(1.0, abs(float(a2[0])))
        rel_err.append(abs(float(a1[0]) - fd1) / s1)
        rel_err.append(abs(float(a2[0]) - fd2) / s2)

    rel_err = np.asarray(rel_err)
    max_err = float(np.max(rel_err))
    med_err = float(np.median(rel_err))
    if max_err > 5.0e-5:
        raise AssertionError(
            f"Gradient consistency failed: max relative error={max_err:.3e}, median={med_err:.3e}"
        )
    print(f"[OK] alpha=grad(phi): max rel. err={max_err:.3e}, median={med_err:.3e}")


def check_circular_isothermal_limit(co):
    """For q=1 and t=1, alpha magnitude should be ~theta_E."""
    theta_E = 1.4
    model = epl(
        co,
        zl=0.5,
        zs=2.0,
        theta_E=theta_E,
        q=1.0,
        pa=0.0,
        t=1.0,
        theta_c=0.0,
    )

    rng = np.random.default_rng(7)
    pts = rng.uniform(-2.2, 2.2, size=(1500, 2))
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[r > 0.4]

    a1, a2 = model.angle(pts[:, 0], pts[:, 1])
    alpha_mag = np.hypot(a1, a2)
    rel = np.abs(alpha_mag - theta_E) / theta_E

    med_rel = float(np.median(rel))
    max_rel = float(np.max(rel))
    if max_rel > 2.0e-10:
        raise AssertionError(
            f"Circular-isothermal limit failed: max rel. err={max_rel:.3e}, median={med_rel:.3e}"
        )

    kappa = model.kappa(pts[:, 0], pts[:, 1])
    expected_kappa = 0.5 * theta_E / np.hypot(pts[:, 0], pts[:, 1])
    kappa_rel = np.abs((kappa - expected_kappa) / expected_kappa)
    max_kappa_rel = float(np.max(kappa_rel))
    if max_kappa_rel > 2.0e-10:
        raise AssertionError(
            f"Circular-isothermal kappa failed: max rel. err={max_kappa_rel:.3e}"
        )

    print(f"[OK] circular t=1 limit: alpha max rel. err={max_rel:.3e}")
    print(f"[OK] circular t=1 limit: kappa max rel. err={max_kappa_rel:.3e}")


def _get_tangential_critical_lines(model, label):
    """Compute tangential critical lines with a consistent error message."""
    try:
        lines = model.tancl(size_principale=0.0)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compute critical lines for {label} with tancl(). "
            "Please ensure scikit-image is installed and working."
        ) from exc
    if len(lines) == 0:
        raise AssertionError(f"No tangential critical lines found for {label}.")
    return lines


def build_map_and_critical_lines(co, save_figure=True, show_figure=False):
    """
    Build EPL diagnostic maps and a SIE-vs-EPL comparison panel.

    Output figure layout:
    - Row 1: EPL diagnostics (kappa, detA, |alpha|) for an example EPL model.
    - Row 2: SIE vs EPL(t=1, q=0.8) comparison maps with both critical-line sets.
    """
    model = epl(
        co,
        zl=0.5,
        zs=2.0,
        theta_E=1.4,
        q=0.7,
        pa=0.3,
        t=1.0,
        theta_c=1.0e-4,
    )

    npix = 350
    fov = 10.0
    theta = np.linspace(-fov / 2.0, fov / 2.0, npix)
    model.setGrid(theta=theta)

    if model.ka.shape != (npix, npix):
        raise AssertionError(f"Unexpected ka shape: {model.ka.shape}")
    if not np.all(np.isfinite(model.ka)):
        raise AssertionError("Non-finite values found in kappa map.")
    if not np.all(np.isfinite(model.a1)) or not np.all(np.isfinite(model.a2)):
        raise AssertionError("Non-finite values found in deflection maps.")
    if not np.all(np.isfinite(model.g1)) or not np.all(np.isfinite(model.g2)):
        raise AssertionError("Non-finite values found in shear maps.")

    tan_lines = _get_tangential_critical_lines(model, "EPL diagnostic setup")
    print(f"[OK] found {len(tan_lines)} tangential critical line(s).")

    detA = (1.0 - model.ka) ** 2 - model.g1**2 - model.g2**2
    alpha_mag = np.hypot(model.a1, model.a2)

    # SIE vs EPL(t=1) comparison setup requested by user.
    cmp_params = dict(
        zl=0.5,
        zs=3.0,
        sigma0=200.0,
        q=0.8,
        pa=0.0,
        theta_c=0.0,
        x1=0.0,
        x2=0.0,
    )
    lens_sie = sie(co, **cmp_params)
    lens_epl = epl(co, **cmp_params, t=1.0)
    lens_sie.setGrid(theta=theta)
    lens_epl.setGrid(theta=theta)
    tan_sie = _get_tangential_critical_lines(lens_sie, "SIE comparison setup")
    tan_epl = _get_tangential_critical_lines(lens_epl, "EPL comparison setup")

    rel_kappa = (lens_epl.ka - lens_sie.ka) / np.maximum(np.abs(lens_sie.ka), 1.0e-12)
    rel_kappa = np.nan_to_num(rel_kappa)

    if save_figure or show_figure:
        extent = [theta.min(), theta.max(), theta.min(), theta.max()]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

        # Row 1: EPL diagnostics.
        im0 = axes[0, 0].imshow(model.ka, origin="lower", extent=extent, cmap="magma")
        for cl in tan_lines:
            xcl, ycl = model.getCritPoints(cl)
            axes[0, 0].plot(xcl, ycl, "-", color="cyan", lw=1.0)
        axes[0, 0].set_title("EPL Convergence (kappa) + Critical Lines")
        axes[0, 0].set_xlabel("theta1 [arcsec]")
        axes[0, 0].set_ylabel("theta2 [arcsec]")
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        lim = np.nanpercentile(np.abs(detA), 99)
        im1 = axes[0, 1].imshow(
            detA,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
        )
        axes[0, 1].contour(model.theta1, model.theta2, detA, levels=[0.0], colors="k", linewidths=1.0)
        axes[0, 1].set_title("Jacobian Determinant det(A)")
        axes[0, 1].set_xlabel("theta1 [arcsec]")
        axes[0, 1].set_ylabel("theta2 [arcsec]")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(
            np.log10(np.clip(alpha_mag, 1.0e-8, None)),
            origin="lower",
            extent=extent,
            cmap="viridis",
        )
        for cl in tan_lines:
            xcl, ycl = model.getCritPoints(cl)
            axes[0, 2].plot(xcl, ycl, "-", color="white", lw=0.8)
        axes[0, 2].set_title("log10(|alpha|)")
        axes[0, 2].set_xlabel("theta1 [arcsec]")
        axes[0, 2].set_ylabel("theta2 [arcsec]")
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, label="log10(arcsec)")

        # Row 2: SIE vs EPL(t=1, q=0.8).
        im3 = axes[1, 0].imshow(lens_sie.ka, origin="lower", extent=extent, cmap="magma")
        for i, cl in enumerate(tan_sie):
            xcl, ycl = lens_sie.getCritPoints(cl)
            axes[1, 0].plot(xcl, ycl, "-", color="cyan", lw=1.1, label="SIE crit." if i == 0 else None)
        for i, cl in enumerate(tan_epl):
            xcl, ycl = lens_epl.getCritPoints(cl)
            axes[1, 0].plot(
                xcl,
                ycl,
                "--",
                color="magenta",
                lw=1.0,
                label="EPL(t=1) crit." if i == 0 else None,
            )
        axes[1, 0].set_title("SIE kappa + SIE/EPL Critical Lines")
        axes[1, 0].set_xlabel("theta1 [arcsec]")
        axes[1, 0].set_ylabel("theta2 [arcsec]")
        axes[1, 0].legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im4 = axes[1, 1].imshow(lens_epl.ka, origin="lower", extent=extent, cmap="magma")
        for cl in tan_sie:
            xcl, ycl = lens_sie.getCritPoints(cl)
            axes[1, 1].plot(xcl, ycl, "-", color="cyan", lw=1.1)
        for cl in tan_epl:
            xcl, ycl = lens_epl.getCritPoints(cl)
            axes[1, 1].plot(xcl, ycl, "--", color="magenta", lw=1.0)
        axes[1, 1].set_title("EPL(t=1) kappa + SIE/EPL Critical Lines")
        axes[1, 1].set_xlabel("theta1 [arcsec]")
        axes[1, 1].set_ylabel("theta2 [arcsec]")
        fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        lim_rel = np.nanpercentile(np.abs(rel_kappa), 99)
        lim_rel = max(lim_rel, 1.0e-12)
        im5 = axes[1, 2].imshow(
            rel_kappa,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-lim_rel,
            vmax=lim_rel,
        )
        for cl in tan_sie:
            xcl, ycl = lens_sie.getCritPoints(cl)
            axes[1, 2].plot(xcl, ycl, "-", color="cyan", lw=0.8)
        for cl in tan_epl:
            xcl, ycl = lens_epl.getCritPoints(cl)
            axes[1, 2].plot(xcl, ycl, "--", color="magenta", lw=0.8)
        axes[1, 2].set_title(r"(EPL - SIE) / SIE  (kappa)")
        axes[1, 2].set_xlabel("theta1 [arcsec]")
        axes[1, 2].set_ylabel("theta2 [arcsec]")
        fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        outpath = Path(__file__).resolve().parent / "test_epl_summary.png"
        if save_figure:
            fig.savefig(outpath, dpi=180)
            print(f"[OK] saved figure: {outpath.name}")
        if show_figure:
            plt.show()
        else:
            plt.close(fig)


def _relative_stats(a_ref, a_test, mask, eps=1.0e-12):
    rel = np.abs(a_test - a_ref) / np.maximum(np.abs(a_ref), eps)
    rel = rel[mask]
    return float(np.median(rel)), float(np.percentile(rel, 95.0)), float(np.max(rel))


def check_sie_epl_t1_equivalence(co):
    """
    Compare SIE and EPL(t=1) using the requested setup:
        sigma0=200 km/s, q=0.8, zl=0.5, zs=3.0
    """
    params = dict(
        zl=0.5,
        zs=3.0,
        sigma0=200.0,
        q=0.8,
        pa=0.0,
        theta_c=0.0,
        x1=0.0,
        x2=0.0,
    )

    lens_sie = sie(co, **params)
    lens_epl = epl(co, **params, t=1.0)

    # In the Tessore-Metcalf convention, theta_E is circularized:
    # theta_E(EPL) = bsie(SIE) * sqrt(q)
    theta_e_from_sie = lens_sie.bsie() * np.sqrt(params["q"])
    if not np.isclose(theta_e_from_sie, lens_epl.theta_E, rtol=1.0e-12, atol=0.0):
        raise AssertionError("SIE and EPL normalizations do not match.")

    rng = np.random.default_rng(2026)
    pts = rng.uniform(-3.0, 3.0, size=(6000, 2))
    rr = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[rr > 0.3]
    x, y = pts[:, 0], pts[:, 1]
    mask = np.ones_like(x, dtype=bool)

    k_sie = lens_sie.kappa(x, y)
    k_epl = lens_epl.kappa(x, y)
    k_med, k_p95, k_max = _relative_stats(k_sie, k_epl, mask)

    a1_sie, a2_sie = lens_sie.angle(x, y)
    a1_epl, a2_epl = lens_epl.angle(x, y)
    a_sie = np.hypot(a1_sie, a2_sie)
    a_epl = np.hypot(a1_epl, a2_epl)
    a_med, a_p95, a_max = _relative_stats(a_sie, a_epl, mask)

    print("[INFO] SIE vs EPL(t=1), q=0.8")
    print(f"       kappa rel. diff: median={k_med:.3e}, p95={k_p95:.3e}, max={k_max:.3e}")
    print(f"       |alpha| rel. diff: median={a_med:.3e}, p95={a_p95:.3e}, max={a_max:.3e}")

    if k_max > 5.0e-6 or a_max > 5.0e-6:
        raise AssertionError(
            f"SIE/EPL(t=1) mismatch too large for q=0.8: "
            f"kappa max rel={k_max:.3e}, |alpha| max rel={a_max:.3e}"
        )
    print("[OK] SIE and EPL(t=1) are equivalent within numerical precision for q=0.8.")


def main():
    parser = argparse.ArgumentParser(description="Test the pyLensLib EPL lens model.")
    parser.add_argument("--show", action="store_true", help="Show diagnostic figure interactively.")
    args = parser.parse_args()

    co = FlatLambdaCDM(H0=70.0, Om0=0.3)

    print("Test 1: potential gradient consistency")
    check_gradient_consistency(co)

    print("Test 2: circular isothermal limit")
    check_circular_isothermal_limit(co)

    print("Test 3: SIE vs EPL(t=1) comparison")
    check_sie_epl_t1_equivalence(co)

    print("Test 4: setGrid + critical lines + diagnostics figure")
    build_map_and_critical_lines(co, save_figure=True, show_figure=args.show)

    print("All EPL tests passed.")


if __name__ == "__main__":
    main()
