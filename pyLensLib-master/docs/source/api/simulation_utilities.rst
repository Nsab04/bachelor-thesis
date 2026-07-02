Simulation Utilities
====================

The ``pyLensLib.cluster.cluster`` class now exposes two SPH map builders:

- ``massMapSPH``: default entry point. Uses legacy ``sphviewer`` when
  available, or the compatible ``sphviewer2`` API from ``py-sphviewer2`` when
  the legacy package is not installed.
- ``massMapSPH_internal``: internal implementation used automatically when
  neither external backend is installed (and callable explicitly).

External SPH backends
---------------------

``massMapSPH`` keeps the historical ``sphviewer.tools.QuickView`` path as the
first choice for backward compatibility. If that import fails, pyLensLib tries
``sphviewer2`` from ``py-sphviewer2``. The new backend does not expose
``QuickView`` directly, so pyLensLib estimates smoothing lengths internally and
passes them to the ``sphviewer2`` renderer. The selected backend is exposed as
``pyLensLib.cluster._SPHVIEWER_BACKEND`` with values ``"sphviewer"``,
``"sphviewer2"``, or ``"internal"``.

Internal SPH kernels
--------------------

``massMapSPH_internal`` supports an explicit ``kernel`` argument:

- ``kernel="cubic_spline"`` keeps the historical internal behavior and remains
  the default.
- ``kernel="wendland_c2"`` uses a compact-support 2D Wendland-C2 kernel with
  the SWIFT/SWIFTSIMIO compact-support convention
  ``kernel_gamma=1.936492`` by default.

Both kernels use the same k-nearest-neighbor smoothing-length estimate and the
same local/global mass-conservation controls. The Wendland-C2 option is useful
for experiments where a smoother, non-negative SPH kernel may reduce
particle-discreteness noise in projected mass maps and in derived lensing maps.
Set ``kernel_gamma`` explicitly to test another Wendland-C2 support radius;
``kernel_gamma=2.0`` reproduces the earlier ``2h`` support convention. The
cubic-spline kernel always uses its historical ``2h`` support.
Because the kernel shape differs from the cubic spline, critical lines, caustics,
and cross sections should be revalidated before using it for production
measurements.

.. automodule:: pyLensLib.cluster
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.gadget
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.perlin
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.perlin_parall
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.samplers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.maputils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.subfind
   :members:
   :undoc-members:
   :show-inheritance:
