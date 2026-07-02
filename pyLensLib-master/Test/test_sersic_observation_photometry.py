"""
Simulate a Sersic source observation and recover its flux.

The source flux used here is the total detector count rate passed to
``pyLensLib.sersic_numba.sersic``. For example, ``source_flux=2000`` and
``exptime=1000`` produces an unlensed source model with 2.0e6 total analytic
counts before truncation by the finite image frame and before adding sky/noise.

Optional lensing magnification is modeled as a local linear transformation:
source-plane coordinates are obtained from image-plane coordinates through a
Jacobian with eigenvalues 1/mu_tangential and 1/mu_radial. This stretches the
image by the tangential and radial magnification factors while conserving
surface brightness.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from astropy import units as u

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyLensLib.sersic_numba import sersic as SersicNumba


@dataclass(frozen=True)
class SimulationConfig:
    image_shape: tuple[int, int] = (192, 192)
    r_eff_arcsec: float = 0.35
    sersic_index: float = 2.0
    source_flux: float = 2000.0
    zeropoint: float = 25.0
    pixel_scale: float = 0.05
    exptime: float = 1000.0
    sky_sb: float = 22.0
    aperture_radius_arcsec: float | None = None
    aperture_radius_reff: float = 4.0
    background_inner_reff: float = 6.0
    background_outer_reff: float = 9.0
    background_inner_aperture_factor: float = 1.25
    background_outer_aperture_factor: float = 1.75
    background_method: str = "input_sky"
    mu_tangential: float = 1.0
    mu_radial: float = 1.0
    magnification_angle: float = 0.0
    oversample: int = 5
    seed: int = 12345


def mag_to_count_rate(mag: float, zeropoint: float) -> float:
    """Return count rate for an object magnitude and detector zero point."""
    return 10.0 ** (-0.4 * (mag - zeropoint))


def sky_sb_to_counts_per_pixel(
    sky_sb: float,
    zeropoint: float,
    pixel_scale: float,
    exptime: float,
) -> float:
    """Convert sky surface brightness from mag/arcsec^2 to counts/pixel."""
    sky_rate_per_arcsec2 = mag_to_count_rate(sky_sb, zeropoint)
    pixel_area_arcsec2 = pixel_scale**2
    return sky_rate_per_arcsec2 * pixel_area_arcsec2 * exptime


def circular_fractional_mask(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    radius_pix: float,
    oversample: int = 8,
) -> np.ndarray:
    """Approximate a circular aperture mask with subpixel area fractions."""
    ny, nx = shape
    y_sub = (np.arange(ny * oversample) + 0.5) / oversample - 0.5
    x_sub = (np.arange(nx * oversample) + 0.5) / oversample - 0.5
    xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
    mask_sub = (xx_sub - center_xy[0]) ** 2 + (yy_sub - center_xy[1]) ** 2 <= radius_pix**2
    return mask_sub.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))


def annulus_mask(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    inner_radius_pix: float,
    outer_radius_pix: float,
) -> np.ndarray:
    """Return a pixel-center annulus mask for local background estimation."""
    ny, nx = shape
    yy, xx = np.indices(shape, dtype=float)
    rr = np.hypot(xx - center_xy[0], yy - center_xy[1])
    return (rr >= inner_radius_pix) & (rr <= outer_radius_pix)


def apply_linear_lensing(
    image_x: np.ndarray,
    image_y: np.ndarray,
    mu_tangential: float,
    mu_radial: float,
    angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map image-plane coordinates to source-plane coordinates with a Jacobian."""
    if mu_tangential <= 0.0 or mu_radial <= 0.0:
        raise ValueError("Tangential and radial magnification factors must be positive")

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    theta_t = cos_a * image_x + sin_a * image_y
    theta_r = -sin_a * image_x + cos_a * image_y

    beta_t = theta_t / mu_tangential
    beta_r = theta_r / mu_radial

    source_x = cos_a * beta_t - sin_a * beta_r
    source_y = sin_a * beta_t + cos_a * beta_r
    return source_x, source_y


def lensed_aperture_scale(config: SimulationConfig) -> float:
    """Scale circular measurement radii to enclose the stretched source."""
    return max(config.mu_tangential, config.mu_radial)


def aperture_radius_pix(config: SimulationConfig) -> float:
    """Return the measurement aperture radius in pixels."""
    radius_scale = lensed_aperture_scale(config)
    if config.aperture_radius_arcsec is not None:
        return config.aperture_radius_arcsec / config.pixel_scale
    return config.aperture_radius_reff * config.r_eff_arcsec / config.pixel_scale * radius_scale


def background_annulus_radii_pix(config: SimulationConfig) -> tuple[float, float]:
    """Return local-background annulus radii in pixels."""
    if config.aperture_radius_arcsec is not None:
        aperture_radius = aperture_radius_pix(config)
        return (
            config.background_inner_aperture_factor * aperture_radius,
            config.background_outer_aperture_factor * aperture_radius,
        )

    r_eff_pix = config.r_eff_arcsec / config.pixel_scale
    radius_scale = lensed_aperture_scale(config)
    return (
        config.background_inner_reff * r_eff_pix * radius_scale,
        config.background_outer_reff * r_eff_pix * radius_scale,
    )


def render_sersic_source(
    config: SimulationConfig,
    apply_lensing: bool = True,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Render a possibly magnified Sersic source from pyLensLib.sersic_numba."""
    ny, nx = config.image_shape
    if nx != ny:
        raise ValueError("sersic_numba test currently expects a square image")

    center_xy = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    oversample = config.oversample
    image_size_arcsec = config.pixel_scale * (nx - 1)

    y_sub = ((np.arange(ny * oversample) + 0.5) / oversample - 0.5 - center_xy[1])
    x_sub = ((np.arange(nx * oversample) + 0.5) / oversample - 0.5 - center_xy[0])
    y_sub *= config.pixel_scale
    x_sub *= config.pixel_scale
    xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
    if apply_lensing:
        source_x, source_y = apply_linear_lensing(
            xx_sub,
            yy_sub,
            config.mu_tangential,
            config.mu_radial,
            config.magnification_angle,
        )
    else:
        source_x, source_y = xx_sub, yy_sub

    model = SersicNumba(
        size=image_size_arcsec,
        Npix=nx,
        gl=None,
        save_unlensed=False,
        n=config.sersic_index,
        q=1.0,
        ys1=0.0,
        ys2=0.0,
        pa=0.0,
        re=config.r_eff_arcsec,
        flux=config.source_flux * config.exptime,
        zs=1.0,
    )
    highres_source = model.render_on_grid(
        source_x,
        source_y,
        rmaxf=100.0,
        pixel_area=(config.pixel_scale / oversample) ** 2,
    )
    source = highres_source.reshape(ny, oversample, nx, oversample).sum(axis=(1, 3))
    return source, center_xy


def simulate_observation(config: SimulationConfig) -> dict[str, np.ndarray | float | tuple[float, float]]:
    """Generate noiseless source, sky image, and Poisson-realized observation."""
    rng = np.random.default_rng(config.seed)
    unlensed_source, center_xy = render_sersic_source(config, apply_lensing=False)
    source, _ = render_sersic_source(config, apply_lensing=True)
    sky_counts = sky_sb_to_counts_per_pixel(
        config.sky_sb,
        config.zeropoint,
        config.pixel_scale,
        config.exptime,
    )
    sky = np.full(config.image_shape, sky_counts, dtype=float)
    expected = source + sky
    observed = rng.poisson(expected).astype(float)
    return {
        "unlensed_source": unlensed_source,
        "source": source,
        "sky": sky,
        "expected": expected,
        "observed": observed,
        "sky_counts_per_pixel": sky_counts,
        "center_xy": center_xy,
    }


def measure_aperture_flux(
    image: np.ndarray,
    config: SimulationConfig,
    center_xy: tuple[float, float],
    sky_counts_per_pixel: float,
) -> dict[str, float]:
    """Measure source flux using an aperture and Astropy background statistics."""
    ccd = CCDData(image, unit=u.electron)
    aperture_radius = aperture_radius_pix(config)
    aperture = circular_fractional_mask(config.image_shape, center_xy, aperture_radius)
    background_inner_radius, background_outer_radius = background_annulus_radii_pix(config)

    bkg_mask = annulus_mask(
        config.image_shape,
        center_xy,
        background_inner_radius,
        background_outer_radius,
    )
    if not np.any(bkg_mask):
        raise ValueError(
            "Background annulus contains no pixels. Reduce the aperture radius, "
            "increase npix, or decrease the background annulus factors."
        )
    background_mean, background_median, background_std = sigma_clipped_stats(
        ccd.data[bkg_mask],
        sigma=3.0,
        maxiters=5,
    )
    if config.background_method == "input_sky":
        background_level = sky_counts_per_pixel
    elif config.background_method == "annulus":
        background_level = float(background_mean)
    else:
        raise ValueError("background_method must be 'input_sky' or 'annulus'")

    aperture_sum = float(np.sum(ccd.data * aperture))
    aperture_area = float(np.sum(aperture))
    source_flux = aperture_sum - background_level * aperture_area
    return {
        "aperture_sum": aperture_sum,
        "aperture_area": aperture_area,
        "background_inner_radius_pix": float(background_inner_radius),
        "background_outer_radius_pix": float(background_outer_radius),
        "background_mean": float(background_mean),
        "background_median": float(background_median),
        "background_level_used": float(background_level),
        "background_method": config.background_method,
        "background_std": float(background_std),
        "measured_source_flux": float(source_flux),
    }


def evaluate_observation(
    config: SimulationConfig,
) -> tuple[dict[str, float], dict[str, np.ndarray | float | tuple[float, float]], dict[str, float]]:
    """Run the simulation and collect photometry products."""
    simulation = simulate_observation(config)
    center_xy = simulation["center_xy"]
    measurement = measure_aperture_flux(
        simulation["observed"],
        config,
        center_xy,
        simulation["sky_counts_per_pixel"],
    )

    aperture_radius = aperture_radius_pix(config)
    aperture = circular_fractional_mask(
        config.image_shape,
        center_xy,
        aperture_radius,
    )
    unlensed_aperture_flux = float(np.sum(simulation["unlensed_source"] * aperture))
    true_aperture_flux = float(np.sum(simulation["source"] * aperture))
    measured_flux = measurement["measured_source_flux"]
    fractional_error = (measured_flux - true_aperture_flux) / true_aperture_flux
    true_aperture_magnification = true_aperture_flux / unlensed_aperture_flux
    recovered_aperture_magnification = measured_flux / unlensed_aperture_flux
    linear_magnification = config.mu_tangential * config.mu_radial

    result = {
        "input_total_counts": config.source_flux * config.exptime,
        "linear_magnification": linear_magnification,
        "expected_lensed_total_counts": config.source_flux
        * config.exptime
        * config.mu_tangential
        * config.mu_radial,
        "rendered_total_counts": float(np.sum(simulation["source"])),
        "aperture_radius_arcsec": aperture_radius * config.pixel_scale,
        "aperture_radius_pix": aperture_radius,
        "background_inner_radius_pix": measurement["background_inner_radius_pix"],
        "background_outer_radius_pix": measurement["background_outer_radius_pix"],
        "unlensed_aperture_flux": unlensed_aperture_flux,
        "true_aperture_flux": true_aperture_flux,
        "measured_aperture_flux": measured_flux,
        "true_aperture_magnification": true_aperture_magnification,
        "recovered_aperture_magnification": recovered_aperture_magnification,
        "true_aperture_mu_over_linear_mu": true_aperture_magnification / linear_magnification,
        "recovered_mu_over_linear_mu": recovered_aperture_magnification / linear_magnification,
        "fractional_error": float(fractional_error),
        "sky_counts_per_pixel": float(simulation["sky_counts_per_pixel"]),
        "background_mean": measurement["background_mean"],
        "background_median": measurement["background_median"],
        "background_level_used": measurement["background_level_used"],
        "background_used_minus_input_sky": measurement["background_level_used"]
        - float(simulation["sky_counts_per_pixel"]),
        "background_method": measurement["background_method"],
        "background_std": measurement["background_std"],
    }
    return result, simulation, measurement


def run_test(config: SimulationConfig, tolerance: float = 0.03) -> dict[str, float]:
    """Run the simulation and assert flux recovery against the noiseless aperture."""
    result, _, _ = evaluate_observation(config)
    assert abs(result["fractional_error"]) < tolerance, result
    return result


def plot_observation(
    config: SimulationConfig,
    result: dict[str, float],
    simulation: dict[str, np.ndarray | float | tuple[float, float]],
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Create a three-panel diagnostic figure for source, model, and observation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from astropy.visualization import ImageNormalize, AsinhStretch

    center_xy = simulation["center_xy"]
    center_arcsec = (0.0, 0.0)
    aperture_radius_arcsec = result["aperture_radius_arcsec"]
    panels = [
        ("Unlensed source", simulation["unlensed_source"]),
        ("Lensed source, no noise", simulation["source"]),
        ("Lensed source with noise", simulation["observed"]),
    ]
    ny, nx = config.image_shape
    extent_arcsec = (
        (0.0 - center_xy[0] - 0.5) * config.pixel_scale,
        (nx - center_xy[0] - 0.5) * config.pixel_scale,
        (0.0 - center_xy[1] - 0.5) * config.pixel_scale,
        (ny - center_xy[1] - 0.5) * config.pixel_scale,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (title, image) in zip(axes, panels):
        image = np.asarray(image)
        vmax = np.nanpercentile(image, 99.7)
        if vmax <= 0.0:
            vmax = np.nanmax(image)
        norm = ImageNormalize(vmin=0.0, vmax=vmax, stretch=AsinhStretch())
        im = ax.imshow(image, origin="lower", cmap="magma", norm=norm, extent=extent_arcsec)
        ax.add_patch(
            Circle(
                center_arcsec,
                aperture_radius_arcsec,
                edgecolor="cyan",
                facecolor="none",
                lw=1.6,
            )
        )
        ax.set_title(title)
        ax.set_xlabel("x - x_image [arcsec]")
        ax.set_ylabel("y - y_image [arcsec]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="counts")

    measurement_text = "\n".join(
        [
            f"measured aperture flux = {result['measured_aperture_flux']:.4g}",
            f"unlensed aperture flux = {result['unlensed_aperture_flux']:.4g}",
            f"true aperture flux = {result['true_aperture_flux']:.4g}",
            f"fractional error = {result['fractional_error']:.3g}",
            f"recovered aperture mu = {result['recovered_aperture_magnification']:.4g}",
            f"recovered/input mu = {result['recovered_mu_over_linear_mu']:.4g}",
            f"background used = {result['background_level_used']:.4g} counts/pix",
            f"background method = {result['background_method']}",
            f"mu_t x mu_r = {result['linear_magnification']:.4g}",
        ]
    )
    axes[2].text(
        0.03,
        0.97,
        measurement_text,
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 5},
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def test_sersic_observation_flux_recovery() -> None:
    """Pytest-compatible smoke test for the standalone simulation."""
    run_test(SimulationConfig())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-eff-arcsec", type=float, default=SimulationConfig.r_eff_arcsec)
    parser.add_argument("--sersic-index", type=float, default=SimulationConfig.sersic_index)
    parser.add_argument("--source-flux", type=float, default=SimulationConfig.source_flux)
    parser.add_argument("--zeropoint", type=float, default=SimulationConfig.zeropoint)
    parser.add_argument("--pixel-scale", type=float, default=SimulationConfig.pixel_scale)
    parser.add_argument("--exptime", type=float, default=SimulationConfig.exptime)
    parser.add_argument("--sky-sb", type=float, default=SimulationConfig.sky_sb)
    parser.add_argument(
        "--aperture-radius-arcsec",
        type=float,
        default=SimulationConfig.aperture_radius_arcsec,
        help=(
            "Circular aperture radius in arcsec. If omitted, use "
            "aperture_radius_reff * r_eff_arcsec * max(mu_tangential, mu_radial)."
        ),
    )
    parser.add_argument(
        "--background-inner-aperture-factor",
        type=float,
        default=SimulationConfig.background_inner_aperture_factor,
        help=(
            "Inner sky-annulus radius in aperture-radius units when "
            "--aperture-radius-arcsec is set."
        ),
    )
    parser.add_argument(
        "--background-outer-aperture-factor",
        type=float,
        default=SimulationConfig.background_outer_aperture_factor,
        help=(
            "Outer sky-annulus radius in aperture-radius units when "
            "--aperture-radius-arcsec is set."
        ),
    )
    parser.add_argument(
        "--background-method",
        choices=("input_sky", "annulus"),
        default=SimulationConfig.background_method,
        help=(
            "Background level used for aperture-flux subtraction. input_sky uses "
            "the known simulated sky; annulus uses the sigma-clipped annulus mean."
        ),
    )
    parser.add_argument("--mu-tangential", type=float, default=SimulationConfig.mu_tangential)
    parser.add_argument("--mu-radial", type=float, default=SimulationConfig.mu_radial)
    parser.add_argument(
        "--magnification-angle",
        type=float,
        default=SimulationConfig.magnification_angle,
        help="Tangential axis angle in radians, measured counterclockwise from image x.",
    )
    parser.add_argument("--npix", type=int, default=SimulationConfig.image_shape[0])
    parser.add_argument("--oversample", type=int, default=SimulationConfig.oversample)
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(__file__).with_name("sersic_observation_photometry.png"),
        help="Path for the diagnostic figure PNG.",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip writing the diagnostic figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in an interactive matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        image_shape=(args.npix, args.npix),
        r_eff_arcsec=args.r_eff_arcsec,
        sersic_index=args.sersic_index,
        source_flux=args.source_flux,
        zeropoint=args.zeropoint,
        pixel_scale=args.pixel_scale,
        exptime=args.exptime,
        sky_sb=args.sky_sb,
        aperture_radius_arcsec=args.aperture_radius_arcsec,
        background_inner_aperture_factor=args.background_inner_aperture_factor,
        background_outer_aperture_factor=args.background_outer_aperture_factor,
        background_method=args.background_method,
        mu_tangential=args.mu_tangential,
        mu_radial=args.mu_radial,
        magnification_angle=args.magnification_angle,
        oversample=args.oversample,
        seed=args.seed,
    )
    result, simulation, _ = evaluate_observation(config)
    assert abs(result["fractional_error"]) < args.tolerance, result

    print("Sersic observation photometry test passed")
    for key, value in result.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            print(f"{key}: {value:.8g}")
        else:
            print(f"{key}: {value}")

    if not args.no_figure:
        plot_observation(config, result, simulation, output_path=args.figure, show=args.show)
        print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
