# LTcloner

LTcloner generates synthetic Lenstool cluster lens models from a reference Lenstool `.par` file. It is designed for cloning an existing cluster model into many randomized realizations while preserving the overall structure of the original system.

The app is YAML-driven: the current entry point is `python ltcloner.py <config.yaml>`. The older command-line interface is no longer supported.

## What LTcloner does

For each generated model, LTcloner:

1. Reads a reference Lenstool parameter file.
2. Extracts the main potentials, cluster galaxies, and gas components.
3. Recenters the model onto a new field of view.
4. Measures the reference galaxy radial distribution.
5. Fits or falls back to a projected NFW-like radial profile.
6. Measures the luminosity function and fits a Schechter function.
7. Measures scaling relations between magnitude and galaxy structural parameters.
8. Samples a new set of galaxy magnitudes and regenerates the cluster member catalog.
9. Writes a new Lenstool `.par` file for each realization.
10. Optionally produces lensing outputs, multiple-image catalogs, cross sections, and diagnostic plots.

## Input

### Required YAML keys

- `parfile`: path to the input Lenstool `.par` file
- `nmodels`: number of lens models to generate

### Common optional keys

- `mmin`, `mmax`: magnitude range used when sampling galaxy magnitudes
- `seed`: base random seed
- `fieldsize`: side length of the output field of view in arcsec
- `src_maglim`: intrinsic source magnitude limit
- `img_maglim`: image magnitude limit
- `bcg`: how many BCG galaxies to align with the main potentials
- `bcg_offset`, `bcg_elloffset`, `bcg_paoffset`: BCG offsets applied during generation
- `opening_angle`: controls the angular distribution of main potentials
- `scatter`: scatter applied to scaling relations
- `scatter_sigma`, `scatter_rcut`, `rho_sigma_rcut`: additional scatter controls used by the generator
- `tolerance`: distance tolerance used when placing model components
- `nbinr`: number of radial bins used when measuring the radial galaxy distribution
- `nullify_ellipticity`: set galaxy ellipticities to zero
- `lens`: generate lensed source and image catalogs
- `test`: run diagnostic plots without treating the run as a production output run
- `compute_cs`: compute cross sections and Einstein radii
- `output_dir`: directory for generated outputs
- `rs_fact`: scale factor applied to the subhalo radial distribution scale radius

See `ltcloner_config.yaml` for a canonical example.

## Output

For each model index `i`, LTcloner writes files named with a `simulated_model_###` prefix into `output_dir`.

### Always written

- `simulated_model_###.par`
- `simulated_model_###_clmemb.csv`
- `simulated_model_###.png` or `simulated_model_###_with_images.png` / `simulated_model_###_no_images.png`, depending on the plotting path

### When `test: true`

- `simulated_model_###_radialdistribution.png`
- `simulated_model_###_luminosity_function.png`
- `simulated_model_###_scaling_relations.png`

### When `compute_cs: true`

- `simulated_model_###_cross_sections.csv`

### When `lens: true`

- `simulated_model_###_sources.csv`
- `simulated_model_###_allimages.csv`
- `simulated_model_###_images.csv`

### Other side effects

- `convergence.npy` may be written when the convergence-map plotting path is used.

## Workflow

### 1. Load the reference model

The app reads the following blocks from the input `.par` file:

- `runmode`
- `grille`
- `potentiel`
- `cline`
- `grande`
- `champ`
- `cosmologie`

It then recenters the model onto a new field of view defined by `fieldsize`.

### 2. Measure the reference cluster properties

LTcloner derives the cloned model statistics from the input cluster:

- **Galaxy radial distribution**: measured from galaxy positions and fit with a projected NFW profile when enough galaxies are available
- **Luminosity function**: measured from galaxy magnitudes and fit with a Schechter function
- **Scaling relations**: measured from the galaxy magnitude vs. velocity-dispersion and cut-radius relations

If any measurement cannot be performed, the script falls back to reasonable default values.

### 3. Generate new model realizations

For each requested model:

- magnitudes are resampled from the fitted luminosity function
- new galaxy properties are assigned from the scaling relations
- the main halo structure is preserved from the reference system
- BCG galaxies can be aligned with the main potentials
- a new Lenstool `.par` file is written

The number of galaxies is not an explicit input; it is derived from the fitted luminosity function and the model field area.

### 4. Optional lensing products

If `lens: true`, LTcloner builds source catalogs, ray-traces them, finds multiple images, and writes image catalogs.

If `compute_cs: true`, LTcloner computes cross sections and related lensing summary quantities for each realization.

If `test: true`, LTcloner generates diagnostic plots comparing the reference model against the generated one.

## Configuration example

```yaml
parfile: "path/to/reference.par"
nmodels: 10
seed: 42
fieldsize: 100.0
mmin: 17.0
mmax: 26.0
scatter: 0.5
opening_angle: 180.0
bcg: 1
lens: false
test: false
compute_cs: false
output_dir: "generated_models"
```

Run it with:

```bash
python ltcloner.py config.yaml
```

## Common use cases

### Quick test run

Use a small model count and enable `test: true` to validate the configuration and inspect the diagnostic plots.

### Production run with lensing

Enable `lens: true` when you want source and image catalogs in addition to the cloned Lenstool model files.

### Cluster with multiple BCGs

Increase `bcg` when you want to associate more BCG galaxies with the main potentials.

## Validation helper

`test_ltcloner_yaml.py` is a standalone script that checks YAML loading and default-value behavior. It is useful when you want to confirm a configuration file parses correctly.

## Practical notes

- `fieldsize` changes both the centered field of view and the expected galaxy counts.
- The main halo count is inherited from the reference model.
- `lens` and `compute_cs` can make runs substantially more expensive.
- Use `ltcloner_config.yaml` as the starting point for new runs.

## Related files

- `ltcloner.py` — main runtime script
- `ltcloner_config.yaml` — example configuration
- `LTCLONER_YAML_GUIDE.md` — parameter reference
- `QUICKSTART_YAML.md` — short run guide
- `generate_lens_models_SUMMARY.md` — high-level workflow summary
- `test_ltcloner_yaml.py` — YAML configuration validation helper
