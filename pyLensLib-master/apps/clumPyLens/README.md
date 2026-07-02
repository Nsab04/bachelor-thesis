# clumPyLens

This folder contains the clumpy-galaxy workflow as a self-contained `pyLensLib` app.

## What lives here

- `clumPyLens.py`: the runnable app
- `input/input_file.yaml`: the default full configuration, organized into YAML sections
- `input/`: bundled lensing maps, PSFs, SED tables, and Yggdrasil lookup tables
- `examples/jwst_sw_demo.yaml`: a smaller JWST demo config
- `TUTORIAL.md`: a step-by-step guide to the app and config layout

## Run it

From the `pyLensLib` repo root:

```bash
python apps/clumPyLens/clumPyLens.py
```

The app will use `apps/clumPyLens/input/input_file.yaml` by default.

To run a specific YAML file:

```bash
CLUMPYLEN_INPUT_FILE=/Users/maxmen3/projects/pyLensLib/apps/clumPyLens/examples/jwst_sw_demo.yaml \
python apps/clumPyLens/clumPyLens.py --render-scale 0.01
```

Useful flags:

- `--render-scale`: lowers the internal rendering grid for faster runs
- `--render-memory-mb`: controls chunking for clump rendering
- `--rmaxf`: trims the expensive outer Sersic wings

## YAML layout

The input file is divided into sections such as `paths`, `runtime`, `scene`,
`bulge`, `disk`, `clumps`, `perlin`, and `instrument`. Each parameter is
written as a small mapping with a `value` field and a human-readable
`description` field.

For a full walkthrough of the structure, see [TUTORIAL.md](./TUTORIAL.md).

## What the prompts mean

The app still asks a few questions at runtime:

- create new clumps or reuse the saved catalog
- choose HR or instrument mode
- choose telescope and filter in instrument mode
- regenerate clumps after changing `disk_inclination_deg` if you want the catalog positions to follow the new projected disk

For a quick JWST example, reuse the saved catalog and choose:

- `Instrument`
- `JWST_SW`
- `F200W`

## Output

Results are written under the output paths configured in the YAML file. The default input writes into `apps/clumPyLens/output/`.

## Notes

- YAML is the canonical input format now.
- Relative paths in the YAML are resolved relative to the YAML file itself.
- If you want a faster demo, start from `examples/jwst_sw_demo.yaml`.
- The disk inclination projects the host before clumps are sampled, so regenerate clumps after changing `disk_inclination_deg`.
