# CL Image Finder

Interactive SIE/PIEMD image finder inspired by `apps/LTinteract/lens_interact_qt.py`.

This first implementation uses Qt's native `QGraphicsView` canvas rather than
Matplotlib redraws. The RGB image and overlays live in arcsec scene coordinates,
so pan/zoom changes only the view transform.

## Run

```bash
python apps/CLImageFinder/cl_image_finder.py
```

Useful options:

```bash
python apps/CLImageFinder/cl_image_finder.py --cluster M0416_B22 --nray 512
python apps/CLImageFinder/cl_image_finder.py --prefix /path/to/model_prefix
python apps/CLImageFinder/cl_image_finder.py --rgb /path/to/cluster_rgb.fits --fov 80
```

## Modes

- `Draw lens`: click and drag on the image plane to set the component center,
  Einstein-radius scale, and orientation. The drag distance sets `theta_E`; the
  drag direction sets the visible major-axis PA. The model, critical curves, and
  caustics update on release. The app starts with no mass component; drawing the
  first ellipse creates the first analytic component. The drawn `theta_E` fixes
  the component velocity-dispersion normalization at the current `zs`; later
  changes to `zs` keep that physical normalization and rescale the deflection
  through the lensing-distance ratio.
- `Profile`: choose whether the next or selected analytic component is an `SIE`
  or a `PIEMD`. Components can be mixed in the same model. For `PIEMD`, the
  drawn ellipse still sets the target equivalent `theta_E`, while `theta_t`
  sets the truncation radius.
- `Add component`: add another analytic mass component. Press the button, then
  draw the new ellipse on the image plane. All analytic components are summed
  into the active lens model.
- `Update model`: rebuild the full current model after edits to component
  parameters, `zs`, `zl`, or `Nray`. Changes to `zl`, `zs`, and `Nray` also
  trigger a rebuild when changed through the spin-box arrows.
- `Use Lenstool model`: load the current cluster Lenstool deflection maps and
  add them as a fixed base model underneath the interactive analytic components.
  `Load Lenstool...` can be used to select another `.par` file; the app expects
  matching `_angx.fits`, `_angy.fits`, and optional `_pot.fits` files. The
  maps are treated as normalized at `zs=1`, which is copied into the `zs` box
  when loaded; changing `zs` rescales the Lenstool base model. Loading or using
  a Lenstool model disables cluster-member catalogs and clears any member layer
  that was already loaded; manually added SIE/PIEMD components remain available.
- `Load members...`: load a cluster-member catalog with a `#REFERENCE type RA DEC`
  first line and galaxy rows containing at least `id RA DEC ... mag ...`. The
  app converts RA/DEC to the current scene coordinates and can add the galaxies
  to the model as PIEMDs.
- `Use members`: include the loaded member galaxies in the composite lens model.
  The member controls use Lenstool-style scaling relations:
  `sigma0 = sigma0_0 * 10^(-0.4 * alpha * (mag - mag_0))`,
  `cut = cut_0 * 10^(-0.4 * beta * (mag - mag_0))`, and
  `core = core_0 * 10^(-0.2 * (mag - mag_0))`. Press `Update members` after
  changing the sliders/spin boxes to rebuild the model.
- To edit an existing component, select it by clicking inside its ellipse in
  `Draw lens` mode. The selected ellipse gets the manipulation handles.
- To remove a component, select its ellipse and press Delete or Backspace.
- In `Draw lens` mode, drag inside the selected ellipse to move it. Drag the cyan
  handle on the minor axis to change the axis ratio `q`. Drag the magenta handle
  on the major axis to change both size and orientation. The blue handle marks
  the lens center.
- If source/image predictions are already displayed, editing the model recomputes
  the image positions from the last clicked source or image. For image-click
  predictions, the originally clicked image remains marked.
- `Click image`: click on the image plane to infer the source position and
  predict the full image set.
- `Mark image family`: click manually identified multiple-image positions on
  the image plane. Each point is labeled as `X.y`, where `X` is the source
  family number and `y` is the image number within that family. Press
  `Add family` when the current family is complete. To remove a marker, click it
  to highlight it and press Delete or Backspace.
- `Export families`: writes the added families to a Lenstool-like image catalog.
  The app exports RA/DEC coordinates, using the Lenstool reference coordinate
  from the `.par` file when available and falling back to the RGB FITS WCS.
- `Load observed...`: load an existing Lenstool-format multiple-image catalog
  such as `obs_arcs_S1063.dat`. The images are drawn as a read-only orange
  overlay using their original Lenstool IDs. `Show observed` toggles this overlay
  in the image viewer.
- `Pan/zoom`: drag to pan. Mouse wheel zooms around the mouse location in all
  modes. In any mode, hold Ctrl (or Cmd on macOS) and drag to temporarily pan
  without switching tools.
- `Show source inset`: displays the source plane as a smaller inset next to the
  main image viewer window.
- `Show viewer`: brings back the separate image-viewer window if it has been
  hidden. The app opens two windows by default: a compact controls window and a
  larger image-viewer window.
- `UI panels`: the small button strip above the controls hides or shows Main
  controls, Lens components, Lenstool, Families, and Cluster member galaxies
  independently. Hide unused panels to give the image plane as much vertical
  space as possible. The Lens components panel shows `theta_t` only for PIEMD
  components.

## Notes

- The app uses the existing `pyLensLib.sie.sie`, `genlen` critical-line methods,
  and `pointsrc` image finder.
- If the RGB FITS file is not available, the app opens with a blank image-plane
  background so the SIE workflow can still be tested.
