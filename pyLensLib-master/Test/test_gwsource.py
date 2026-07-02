from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyLensLib.gwlensing import (
    dimensionless_frequency,
    geometric_optics_amplification,
    point_mass_regime,
)
from pyLensLib.gwsource import gwsource


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts_gwsource"


class DummyWeakLens:
    def __init__(self, kappa=0.1, gamma1=0.05, gamma2=0.02):
        self.thetax = np.array([-1.0, 0.0, 1.0])
        self.thetay = np.array([-1.0, 0.0, 1.0])
        self.pixel_scale = 1.0
        self.ka = np.full((3, 3), kappa, dtype=float)
        self.g1 = np.full((3, 3), gamma1, dtype=float)
        self.g2 = np.full((3, 3), gamma2, dtype=float)
        self.zl = 0.5
        self.zs = 2.0


def test_gwsource_generates_unlensed_newtonian_waveform():
    freq = np.array([20.0, 30.0, 40.0])
    src = gwsource(frequencies=freq, lensing="none")

    f, h0 = src.unlensed_waveform()
    f_lensed, h_lensed = src.lensed_waveform()

    np.testing.assert_allclose(f, freq)
    np.testing.assert_allclose(f_lensed, freq)
    assert h0.shape == freq.shape
    assert np.iscomplexobj(h0)
    assert np.all(np.isfinite(h0))
    np.testing.assert_allclose(h_lensed, h0)
    assert src.regime_selected == "none"


def test_gwsource_accepts_user_supplied_frequency_domain_waveform():
    freq = np.array([10.0, 20.0, 30.0])
    h0 = np.array([1.0 + 0.0j, 0.5 + 0.2j, 0.1 - 0.3j])

    src = gwsource(waveform=(freq, h0), lensing="none")

    np.testing.assert_allclose(src.frequencies, freq)
    np.testing.assert_allclose(src.unlensed_waveform()[1], h0)


def test_weak_lensing_scales_strain_by_sqrt_magnification():
    lens = DummyWeakLens(kappa=0.1, gamma1=0.05, gamma2=0.02)
    freq = np.array([20.0, 40.0])
    src = gwsource(frequencies=freq, gl=lens, lensing="weak", regime="weak")

    amp = src.amplification_factor()

    det_a = (1.0 - 0.1) ** 2 - 0.05**2 - 0.02**2
    expected = np.sqrt(abs(1.0 / det_a))
    np.testing.assert_allclose(amp, expected)


def test_point_mass_auto_selects_wave_optics_for_small_lens():
    freq = np.array([20.0, 40.0, 80.0])
    src = gwsource(
        frequencies=freq,
        wave_lens={"model": "point_mass", "mass": 100.0, "y": 0.8, "zl": 0.5},
        lensing="auto",
    )

    amp = src.amplification_factor()

    assert src.regime_selected == "wave"
    assert amp.shape == freq.shape
    assert np.iscomplexobj(amp)
    assert np.all(np.isfinite(amp))


def test_point_mass_auto_selects_geometric_optics_for_large_lens():
    freq = np.array([20.0, 40.0, 80.0])
    src = gwsource(
        frequencies=freq,
        wave_lens={"model": "point_mass", "mass": 1.0e7, "y": 0.8, "zl": 0.5},
        lensing="auto",
    )

    f_lensed, h_lensed = src.lensed_waveform()

    assert src.regime_selected == "geometric"
    np.testing.assert_allclose(f_lensed, freq)
    assert h_lensed.shape == freq.shape
    assert np.all(np.isfinite(h_lensed))


def test_hybrid_uses_macro_weak_scaling_and_compact_wave_factor_when_no_td_surface():
    lens = DummyWeakLens(kappa=0.2, gamma1=0.01, gamma2=0.0)
    freq = np.array([20.0, 40.0])
    src = gwsource(
        frequencies=freq,
        gl=lens,
        wave_lens={"model": "point_mass", "mass": 100.0, "y": 1.0, "zl": 0.5},
        lensing="hybrid",
        regime="hybrid",
    )

    amp = src.amplification_factor()

    det_a = (1.0 - 0.2) ** 2 - 0.01**2
    assert amp.shape == freq.shape
    assert np.all(np.isfinite(amp))
    assert np.all(np.abs(amp) > np.sqrt(abs(1.0 / det_a)) * 0.1)


def test_geometric_optics_amplification_keeps_frequency_shape():
    freq = np.array([10.0, 20.0, 30.0])
    amp = geometric_optics_amplification(
        freq,
        magnifications=np.array([2.0, -0.5]),
        delays=np.array([0.0, 0.01]),
        morse_indices=np.array([0, 1]),
    )

    assert amp.shape == freq.shape
    assert np.iscomplexobj(amp)
    assert np.all(np.isfinite(amp))


def test_point_mass_regime_thresholds():
    freq = np.array([20.0, 40.0])

    assert point_mass_regime(freq, 100.0, zl=0.0) == "wave"
    assert point_mass_regime(freq, 1.0e7, zl=0.0) == "geometric"


def make_gwsource_validation_plots(output_dir=ARTIFACT_DIR):
    """Create validation plots showing the main GW-source lensing modes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    freq = np.linspace(10.0, 250.0, 241)
    src_unlensed = gwsource(frequencies=freq, lensing="none")
    src_weak = gwsource(
        frequencies=freq,
        gl=DummyWeakLens(kappa=0.12, gamma1=0.04, gamma2=0.02),
        lensing="weak",
        regime="weak",
    )
    src_wave = gwsource(
        frequencies=freq,
        wave_lens={"model": "point_mass", "mass": 100.0, "y": 0.8, "zl": 0.5},
        lensing="wave",
        regime="wave",
    )
    src_geo = gwsource(
        frequencies=freq,
        wave_lens={"model": "point_mass", "mass": 1.0e7, "y": 0.8, "zl": 0.5},
        lensing="geometric",
        regime="geometric",
    )

    _, h0 = src_unlensed.unlensed_waveform()
    _, h_weak = src_weak.lensed_waveform()
    _, h_wave = src_wave.lensed_waveform()
    _, h_geo = src_geo.lensed_waveform()

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    norm = np.nanmax(np.abs(h0))
    ax.loglog(freq, np.abs(h0) / norm, label="unlensed", lw=2.0)
    ax.loglog(freq, np.abs(h_weak) / norm, label="weak lensing", lw=1.8)
    ax.loglog(freq, np.abs(h_wave) / norm, label="point-mass wave optics", lw=1.8)
    ax.loglog(freq, np.abs(h_geo) / norm, label="point-mass geometric optics", lw=1.4)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$|h(f)|$ / max unlensed")
    ax.set_title("GW source waveform amplitudes")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    waveform_path = output_dir / "gwsource_waveform_modes.png"
    fig.tight_layout()
    fig.savefig(waveform_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    for label, src in [
        ("weak", src_weak),
        ("wave", src_wave),
        ("geometric", src_geo),
    ]:
        amp = src.amplification_factor()
        ax[0].plot(freq, np.abs(amp), label=label, lw=1.8)
        ax[1].plot(freq, np.unwrap(np.angle(amp)), label=label, lw=1.8)
    ax[0].set_ylabel(r"$|F(f)|$")
    ax[0].set_title("GW lensing amplification factors")
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].set_xlabel("frequency [Hz]")
    ax[1].set_ylabel(r"unwrapped arg $F(f)$ [rad]")
    ax[1].grid(True, alpha=0.25)
    amplification_path = output_dir / "gwsource_amplification_factors.png"
    fig.tight_layout()
    fig.savefig(amplification_path, dpi=180)
    plt.close(fig)

    masses = np.logspace(1.0, 8.0, 180)
    band_freq = np.linspace(10.0, 300.0, 180)
    mass_grid, freq_grid = np.meshgrid(masses, band_freq)
    w_grid = dimensionless_frequency(freq_grid, mass_grid, zl=0.5)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    mesh = ax.pcolormesh(masses, band_freq, np.log10(w_grid), shading="auto", cmap="viridis")
    ax.contour(masses, band_freq, w_grid, levels=[10.0, 100.0], colors=["white", "black"], linewidths=1.5)
    ax.text(1.8e1, 260.0, "wave", color="white", fontsize=10)
    ax.text(8.0e4, 260.0, "hybrid", color="white", fontsize=10)
    ax.text(2.5e7, 260.0, "geometric", color="black", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel(r"compact-lens mass [$M_\odot$]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(r"Point-mass regime diagnostic: $w = 8\pi GM_z f/c^3$")
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(r"$\log_{10} w$")
    regime_path = output_dir / "gwsource_point_mass_regime_map.png"
    fig.tight_layout()
    fig.savefig(regime_path, dpi=180)
    plt.close(fig)

    return {
        "waveform_modes": waveform_path,
        "amplification_factors": amplification_path,
        "regime_map": regime_path,
    }


def test_gwsource_validation_plots_are_created(tmp_path):
    paths = make_gwsource_validation_plots(tmp_path)

    assert set(paths) == {"waveform_modes", "amplification_factors", "regime_map"}
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


if __name__ == "__main__":
    for label, path in make_gwsource_validation_plots().items():
        print(f"{label}: {path}")
