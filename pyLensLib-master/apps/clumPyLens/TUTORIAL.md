# clumPyLens Tutorial

This tutorial walks through the bundled `clumPyLens` app, the YAML input
format, and the common ways to run the example configurations.

## What this app does

`clumPyLens` builds a lensed galaxy with bulge, disk, and optional clump
structure, then writes FITS images, catalogues, and diagnostic plots. The
workflow can run in a high-resolution branch or in an instrument branch for
specific telescope and filter combinations.

## Folder guide

- `clumPyLens.py`: the main executable
- `input/input_file.yaml`: the default full configuration
- `examples/jwst_sw_demo.yaml`: a smaller JWST demo
- `input/`: the bundled lensing maps, PSFs, SED tables, and helper inputs
- `output/`: created at runtime for generated products

## Running the app

From the `pyLensLib` repository root:

```bash
python apps/clumPyLens/clumPyLens.py
```

To use the demo config:

```bash
CLUMPYLEN_INPUT_FILE=/Users/maxmen3/projects/pyLensLib/apps/clumPyLens/examples/jwst_sw_demo.yaml \
python apps/clumPyLens/clumPyLens.py --render-scale 0.01
```

Useful flags:

- `--render-scale`: lowers the internal rendering grid for faster runs
- `--render-memory-mb`: adjusts clump chunking based on memory budget
- `--rmaxf`: trims the expensive outer Sersic wings

## YAML structure

The input file is intentionally split into sections so it is easier to scan
and edit. Each parameter is stored as a small mapping with a `value` field and
an optional `description` field.

Example:

```yaml
scene:
  description: Geometry, lensing setup, and image framing.
  pix_scale:
    value: 0.0025
    description: High-resolution pixel scale in arcsec per pixel.
  cluster:
    value: M0416
    description: Cluster name used to select the matching deflection maps.
```

The app reads this as a normal flat configuration internally, so the sections
are for humans, not for extra runtime complexity.

The `disk_inclination_deg` value projects the host before new clumps are sampled, so the clump catalog inherits the same tilt when you create it.
The `perlin` section controls the optional disk texture. Increase the scale to broaden the structures, or raise the octaves to add finer detail.

## Section guide

- `paths`: input and output directories
- `runtime`: interactive yes/no switches
- `scene`: lensing geometry, source placement, and image size
- `bulge`: bulge mass, structure, and reference magnitudes
- `disk`: disk mass, structure, inclination, and reference magnitudes
- `clumps`: clump mass and age distribution controls
- `perlin`: optional disk texture controls such as scale, octaves, persistence, lacunarity, and seed
- `instrument`: exposure times, sky backgrounds, zero points, and pixel scales

## Runtime prompts

The app still asks a few questions when it starts:

- create new clumps or reuse the saved catalog
- choose HR or instrument mode
- choose telescope and filter in instrument mode
- regenerate clumps after changing `disk_inclination_deg` if you want the catalog positions to follow the new projected disk

For a quick JWST example, use:

- `Instrument`
- `JWST_SW`
- `F200W`

## Output locations

Outputs are written to the directories configured in the YAML file. The
default input writes into `apps/clumPyLens/output/`, while the demo writes into
`apps/clumPyLens/output/demo_jwst/`.

## Editing tips

- Keep paths relative to the YAML file that contains them.
- Use the `value` field for the actual parameter data.
- Put short, plain-language explanations in `description`.
- If you need to make a fast smoke-test run, start from the demo YAML and turn
  `--render-scale` down.
