Ray Tracing and Multi-Plane Propagation
=======================================

Mass-map FFT backends
---------------------

``pyLensLib.raytracer.raytracer`` accepts a projected mass map and computes
deflection-angle maps on a requested ray grid. The ``method`` keyword selects
the numerical backend:

``potential_fft``
   Legacy pyLensLib behavior and current default. The convergence map is
   padded, a Fourier Poisson solve produces the lensing potential, the potential
   is interpolated to the ray grid, and finite differences produce ``a1/a2``.

``deflection_fft``
   Direct Green's-function convolution from convergence to deflection angles.
   The potential is still available, but it is computed separately from
   convergence with a logarithmic potential kernel rather than by integrating
   the deflection field.

``deflection_fft_adaptive``
   Split-resolution approximation to ``deflection_fft``. It evaluates the
   near-field contribution on the original grid and the far-field contribution
   on a downsampled grid controlled by ``low_res_factor`` and
   ``high_res_kernel_size``.

Example::

   from pyLensLib.raytracer import raytracer

   rt = raytracer(
       co,
       mass_map,
       Nray=2048,
       FOVray=200.0,
       fromfile=False,
       zl=zl,
       zs=zs,
       fov=400.0,
       method="deflection_fft",
   )

For snapshot comparisons of the three modes, see
``Test/test_compare_snapshot_deflection_fft_modes.py``.

.. automodule:: pyLensLib.raytracer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.convergence_integrals
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.raymesh
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.multiplane
   :members:
   :undoc-members:
   :show-inheritance:
