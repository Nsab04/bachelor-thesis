# FitPIEMDs

FitPIEMDs contains the scripts used to fit LensTool PIEMD galaxy profiles and
compare the resulting dark-matter-plus-stellar decompositions against the
cluster-member catalog for the M0416-style data in this folder.

The two primary workflows are:

- `fitGalaxies_tnfw_jaffe_dual.py`
- `fit_bergamini_bmo_free_jaffe_grid.py`

A secondary utility, `measure_mass_radial_profile.py`, measures annular mass
surface density profiles from Lenstool `mass.fits` maps.

## 1) `fitGalaxies_tnfw_jaffe_dual.py`

This is the main galaxy-fitting script.
It fits each selected LensTool galaxy with a total density model made from:

- a dark-matter profile
- a Jaffe stellar component

It supports multiple dark-matter families:

- truncated NFW (`tNFW`, default)
- BMO (`--use-bmo`)
- standard NFW (`--use-nfw`)
- Einasto (`--use-einasto`)
- generalized NFW (`--use-gnfw`)

It also supports two stellar configurations:

- **fixed-stellar fit**: Jaffe profile constrained from the photometric estimate
- **free-stellar fit**: stellar mass and Jaffe scale radius both free

### Input files

The script works with the LensTool / catalog files in this directory:

- a reference `.par` file, default `M0416.par`
- the LensTool galaxy catalog / auxiliary catalog files
- the VizieR-style README table used to load the external cluster catalog

### Outputs

The script writes a combined CSV/HDF5 result set with the fitted parameters and
model diagnostics, plus comparison plots. The exact column set depends on the
chosen DM profile family, but the output always includes the fitted galaxy
parameters, fit quality metrics, and derived halo quantities such as `c200`,
`R200`, and `M200` when they are defined for the selected model.

### Typical use

```bash
python fitGalaxies_tnfw_jaffe_dual.py \
  --parfile M0416.par \
  --catalog M0416.tsv \
  --readme ReadMe.txt \
  --out-csv lens_galaxy_fits_tnfw_jaffe_dual.csv
```

### Practical notes

- The script auto-detects the DM profile family from the CLI flags.
- It writes profile-specific output columns, so the CSV layout depends on the
  chosen DM model.
- The companion validation script is `validate_dual_fit_outputs.py`.

## 2) `fit_bergamini_bmo_free_jaffe_grid.py`

This script generates a grid of PIEMD-like galaxy profiles by sampling
`sigma0`, deriving `r_cut(sigma0)` from the reference model, and fitting each
grid point with a **BMO + free Jaffe** decomposition.

It is used to explore the concentration-mass relation implied by the Bergamini
cluster-member scaling.

### What it does

- reads the reference `.par` file and measures the galaxy `r_cut` scaling
- builds a grid in `sigma0`
- constructs a PIEMD density profile for each grid point
- fits that profile with BMO + free Jaffe
- computes derived diagnostics such as `c200`, `R200`, and `M200`
- writes the grid results and diagnostic plots

### Input files

The script defaults to the local `M0416.par` reference file, but it can also use:

- a user-provided `.par` file for the `r_cut(sigma0)` scaling
- the dual-fit CSV for optional stellar-mass initialization and overlays

### Outputs

- `bergamini_bmo_free_jaffe_grid.csv`
- `bergamini_bmo_free_jaffe_c200_vs_M200.png`
- `bergamini_sigma0_vs_rcut.png`
- `bergamini_bmo_free_jaffe_profiles_mosaic_4x4.png`

### Typical use

```bash
python fit_bergamini_bmo_free_jaffe_grid.py \
  --parfile M0416.par \
  --out-csv bergamini_bmo_free_jaffe_grid.csv
```

### Practical notes

- The `r_cut(sigma0)` relation can be derived from the reference `.par` file or
  supplied explicitly on the CLI.
- The script uses `M0416_B19.par`-style Bergamini assumptions by default,
  including the vanishing-core PIEMD setup when `--r-core-kpc 0` is used.
- The grid output is useful for checking the implied `c200-M200` trend and for
  comparing against the dual-fit results.

## 3) `measure_mass_radial_profile.py`

This helper measures annular mass surface density profiles from a Lenstool
`mass.fits` map.
It is mainly a diagnostic / analysis utility rather than a fitting pipeline.

### What it measures

- pixels are binned into linear or logarithmic radial annuli
- the total mass in each annulus is summed
- the total is divided by the annulus area
- the radius is reported in kpc
- the profile is plotted in `M$_\odot$/kpc$^2$`

For the Lenstool maps used here, `mass.fits` values are treated as
`10^12 M$_\odot$/pixel`.

### Output

- a CSV table with radius, annulus mass, area, and surface density columns
- a PNG plot with kpc on the x-axis and mass surface density on the y-axis

### Example

```bash
python measure_mass_radial_profile.py --ra 64.0391 --dec -24.06775 --logspace
```

## Common conventions in this folder

- LensTool galaxy profiles use the PIEMD / Jaffe conventions from the fitting
  scripts in this directory.
- The reference model files (`M0416.par`, `M0416.tsv`, `M0416_B19.par`) are
  the starting point for the main workflows.
- Most outputs are written next to the scripts unless you override the output
  paths on the CLI.

## Related files

- `validate_dual_fit_outputs.py` — checks the main dual-fit outputs
- `M0416.par` — reference LensTool model
- `M0416.tsv` — catalog used by the fitting workflow
- `M0416_B19.par` — Bergamini-style reference model
- `ReadMe.txt` / catalog tables — external catalog metadata used by the fit scripts
