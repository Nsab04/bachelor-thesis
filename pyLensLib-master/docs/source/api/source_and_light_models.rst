Source and Light Models
=======================

This section includes both:

- source-model classes inheriting from ``pyLensLib.gensrc.gensrc`` (e.g.,
  ``sersic``, ``pointsrc``, ``shapelets``, ``poststamp``), and
- related source/light utilities that do not inherit from ``gensrc``
  (e.g., ``spiral`` and ``sourceplane``).

The ``pointsrc`` model now also supports optional time-delay-surface refinement
of image positions via ``pointsrc(..., refine_to_td=True)``. When enabled, the
raw image candidates found by the triangle-mapping solver are snapped to the
nearest stationary points of the plotted time-delay surface.

``gwsource`` adds frequency-domain gravitational-wave sources.  It can produce
a lightweight built-in inspiral waveform or accept an external waveform, then
apply weak lensing, geometric-optics strong lensing, point-mass wave optics, or
a macro-lens plus compact-lens hybrid amplification.

.. automodule:: pyLensLib.sersic
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.sersic_numba
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.pointsrc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.gwsource
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.gwlensing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.shapelets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.shapelets_polar
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.spiral
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.sourceplane
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.poststamp
   :members:
   :undoc-members:
   :show-inheritance:
