# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is
`pyLensLib` is a scientific Python package for gravitational lensing simulations across microlensing, galaxy-scale lensing, and cluster-scale strong lensing / GGSL workflows.

## Common commands
- Install package: `pip install .`
- Editable dev install: `pip install -e .`
- Install dependencies: `pip install -r requirements.txt`
- Build docs: `PYTHONPATH=$(pwd) make -C docs clean html`
- Run one validation script: `python Test/<script>.py` (for example, one of the `Test/test_*.py` scripts)

## Architecture at a glance
- `genlen` is the base class for lens models. It owns grid setup plus critical-line / caustic computation.
- `gensrc` is the base class for source models. It handles ray tracing.
- Lens models split into two broad groups:
  - analytical models such as `piemd`, `sie`, `epl`, `pepl`, `nfwell`, and `extshear`
  - map-based deflectors via `deflector`, which load precomputed deflection-angle grids
- Source models include `sersic`, `sersic_numba`, and `pointsrc`.
- `observation` handles PSF / noise / photometric effects.
- `critcau` stores critical-line and caustic geometry.
- `apps/` contains batch or production workflows, `Test/` contains end-to-end validation scripts and notebooks, and `docs/` is the Sphinx documentation tree.

## Repo conventions worth keeping in mind
- Use the established coordinate names consistently: image plane `theta1/theta2`, source plane `y1/y2` or `ys1/ys2`, deflection angles `angx/angy`.
- For direct lens mapping, deflector and source grids usually need matching `fov` and `npix`.
- Multi-redshift sources rescale deflections automatically.
- Use `sersic_numba` for performance-sensitive runs and `sersic` when debugging.
- Optional dependencies matter for some modules; check `requirements.txt` and the README before assuming a feature is available.
