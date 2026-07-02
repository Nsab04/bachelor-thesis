Core Architecture
=================

Class Hierarchy and Inheritance
-------------------------------

Lens Models
~~~~~~~~~~~

Base class
^^^^^^^^^^

- ``genlen`` (``pyLensLib/genlen.py``):
  Base class for lens models. Handles grid setup, critical-line search,
  and caustic computation.

Classes inheriting from ``genlen``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``deflector``: loads pre-computed deflection angle maps from FITS files.
- ``piemd``: analytical Pseudo-Isothermal Elliptical Mass Distribution.
- ``sie``: analytical Singular Isothermal Ellipsoid.
- ``nfwell``: analytical elliptical NFW-based lens model.
- ``epl``: Elliptical Power-Law model (Tessore & Metcalf formulation).
- ``pepl``: pseudo-elliptical power-law model (legacy pyLensLib formulation).
- ``multipoles``: multi-pole lens perturbation model.
- ``compositeModel``: additive combination of multiple lens models.

Classes not inheriting from ``genlen``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``ext_shear`` (``pyLensLib/extshear.py``): external shear utility model.
- ``perturber`` (``pyLensLib/perturber.py``): local lens perturbation helper.
- ``CriticalLine`` and ``Caustic`` (``pyLensLib/critcau.py``):
  Geometry containers for critical lines and caustics returned by lens models.

Microlensing
~~~~~~~~~~~~

- ``point_source`` (``pyLensLib/microlensing.py``): source model for
  microlensing light-curve calculations.
- ``point_lens`` (``pyLensLib/microlensing.py``): single point-mass lens model.
- ``binary_lens`` (``pyLensLib/microlensing.py``): binary point-mass lens model.

Source Models
~~~~~~~~~~~~~

Base class
^^^^^^^^^^

- ``gensrc`` (``pyLensLib/gensrc.py``):
  Base class for source models. Handles ray tracing and image generation hooks.

Classes inheriting from ``gensrc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``sersic``: extended source with a Sersic profile.
- ``sersic_numba``: Numba-optimized Sersic implementation.
- ``pointsrc``: point-source model with image-finding utilities, including
  optional refinement to time-delay-surface stationary points.
- ``gwsource``: frequency-domain gravitational-wave source model.  It supports
  unlensed waveform generation or user-supplied waveforms, weak lensing,
  geometric-optics strong lensing through existing point-source image finding
  and time-delay surfaces, point-mass wave optics for compact lenses, and a
  macro-lens plus compact-lens hybrid mode.
- ``shapelets`` and ``shapelets_polar``: shapelet-based source representations.
- ``poststamp``: cutout-based source/light handling utilities for image-plane workflows.

Classes not inheriting from ``gensrc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``spiral`` (``pyLensLib/spiral.py``): multiplicative spiral-structure model
  applied to an existing disc-like source model.
- ``sersic_sourceplane`` (``pyLensLib/sourceplane.py``): helper class to build
  source-plane populations from multiple Sersic components.

Ray Tracing and Multi-Plane Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``raytracer`` (``pyLensLib/raytracer.py``): grid-based ray-tracing engine.
  It converts projected mass maps to convergence and then computes lensing
  potential and deflection-angle maps. The default ``method="potential_fft"``
  preserves the historical pyLensLib path, while ``method="deflection_fft"``
  and ``method="deflection_fft_adaptive"`` use direct convergence-to-deflection
  FFT convolution backends.
- ``convergence_integrals`` (``pyLensLib/convergence_integrals.py``):
  standalone Green's-function convolution helpers for deriving deflection
  angles and potentials from convergence or surface-density maps.
- ``raymesh`` (``pyLensLib/raymesh.py``): ray-mesh helper for interpolation.
- ``lensplane`` (``pyLensLib/multiplane.py``): single lens-plane container.
- ``multiplane`` (``pyLensLib/multiplane.py``): multi-plane lensing propagation.

Mass-map FFT backends
^^^^^^^^^^^^^^^^^^^^^

The three ``raytracer`` backends differ in the discretized operation used after
the mass map has been converted to convergence:

- ``potential_fft``: pads the convergence map, solves the Poisson equation for
  the potential in Fourier space, interpolates the potential to the ray grid,
  and finite-differences the interpolated potential to obtain ``a1/a2``.
- ``deflection_fft``: directly convolves convergence with the two deflection
  kernels to obtain ``a1/a2``. The potential is computed separately with the
  logarithmic potential kernel.
- ``deflection_fft_adaptive``: approximates ``deflection_fft`` by combining a
  high-resolution near-field contribution with a low-resolution far-field
  contribution. It is useful for performance experiments on large maps, but
  should be validated against ``deflection_fft`` for the target science metric.

Observational Classes
~~~~~~~~~~~~~~~~~~~~~

- ``observation``:
  Handles PSF convolution, photon noise, background, and magnitude/counts
  conversions.

Simulation Utilities
~~~~~~~~~~~~~~~~~~~~

- ``cluster`` (``pyLensLib/cluster.py``):
  Utilities for cluster-scale simulation workflows and data handling. The
  internal SPH mass-map builder supports both the historical compact cubic
  spline and an opt-in Wendland-C2 kernel. Wendland-C2 defaults to the
  SWIFT/SWIFTSIMIO support factor ``kernel_gamma=1.936492``.
- ``snapshot_header`` (``pyLensLib/gadget.py``):
  Gadget snapshot-header reader.
- ``perlin`` and ``perlin_parall``:
  Perlin-noise generators used by simulation workflows.
- ``samplers``: sampling utilities for synthetic populations.
- ``map_obj``, ``contour_fit``, ``image_fit`` (``pyLensLib/maputils.py``):
  map handling and fitting helpers.
- ``subfind`` (``pyLensLib/subfind.py``): SUBFIND catalog I/O helpers.

Catalogs, Populations, and External Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``srccatalog`` (``pyLensLib/srccatalog.py``): synthetic source-catalog builder.
- ``photocatalogs`` (module): photometric catalog helpers.
- ``sedmodel`` and ``clusterSEDs`` (``pyLensLib/sedmodel.py``):
  spectral-energy-distribution utilities.
- ``Passband`` and ``SED`` (``pyLensLib/sedcompat.py``):
  local replacements for legacy SED/passband utilities.
- ``subhaloPop`` (``pyLensLib/subhaloPop.py``): subhalo population generators.
- ``lenstool`` (``pyLensLib/lenstool.py``): Lenstool-style input/output helpers.

Documentation Groups
--------------------

The API reference is organized into these groups:

1. Core Foundations
2. Lens Mass Models
3. Source and Light Models
4. Ray Tracing and Multi-Plane Propagation
5. Simulation Utilities
6. Catalogs, Populations, and External Data
7. Observation and Instrument Effects

For full module/class API details, see :doc:`modules`.
