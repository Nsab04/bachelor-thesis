Core Foundations
================

Optional geometry dependencies
------------------------------

Most critical-line and caustic operations work with ``shapely`` alone.
``geopandas`` is optional, but still required by methods that sample or select
points inside caustic geometries, including
``Caustic.random_points_in_caustic()``,
``Caustic.random_points_along_caustic()``, and
``genlen.select_points_in_caustics()``.

.. automodule:: pyLensLib
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.genlen
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.gensrc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyLensLib.critcau
   :members:
   :undoc-members:
   :show-inheritance:
