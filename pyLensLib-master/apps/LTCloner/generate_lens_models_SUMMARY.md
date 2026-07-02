# generate_lens_models.py - Code Summary

## Purpose
Generates synthetic strong gravitational lensing models for galaxy clusters, creating realistic mock observations for testing lensing analysis pipelines.

## Input/Output

### Input
- Reference Lenstool `.par` file containing a cluster lens model with galaxy distributions

### Output
- Multiple randomized cluster lens models with:
  - Modified galaxy catalogs matching observed scaling relations
  - Lensed source catalogs with multiple images
  - Full Lenstool-compatible parameter files
  - Optional mock observations with convergence maps and critical lines

## Key Processing Steps

### 1. Model Analysis
- Reads reference cluster model from Lenstool `.par` file
- Extracts main potentials (dark matter halos with `id` starting with 'O'), member galaxies (potentials with `mag` keyword), and gas (remaining potentials)
- Measures radial distribution (fits NFW profile: `n₀`, `rs`)
- Measures luminosity function (fits Schechter parameters: `φ*`, `M*`, `α`)
- Measures scaling relations: velocity dispersion vs magnitude (`α`), cut radius vs magnitude (`β`)

### 2. Galaxy Generation
- **Number of galaxies**: Determined automatically by integrating the Schechter luminosity function over the magnitude range [mmin, mmax] and multiplying by the field area (in arcmin²):
  ```
  N_expected = ∫(mag_min to mag_max) Φ(m) dm × Area
  ```
  where Φ(m) is the Schechter function fitted from the reference model
- Samples galaxy magnitudes from fitted Schechter function using inverse transform sampling
- Assigns velocity dispersions and cut radii using scaling relations with lognormal scatter
- Positions galaxies following NFW radial profile
- Optional: forces BCG alignment with main dark matter potentials

### 3. Lensing Simulation (if `--lens` flag set)
- Generates background source catalog using `srccatalog` (real galaxy properties from HUDF)
- Ray-traces sources through deflector using `pointsrc` for multi-plane lensing
- Finds multiple images via lens equation solver
- Filters images by magnitude limit
- Saves image catalog in Lenstool format
- Creates visualization with convergence map, critical lines, and multiple images overlaid (each source's images shown in a unique color)

### 4. Validation Tests (if `--test` flag set)
- Plots radial distribution vs NFW fit
- Plots luminosity function vs Schechter fit
- Plots scaling relations (v_disp and r_cut vs magnitude)
- Visualizes convergence maps with critical lines/caustics

## Technical Implementation

### Multi-Plane Lensing
- Distributes sources across 100 redshift planes (z_lens to z=11)
- Checks caustic membership to pre-select multiply-imaged sources
- Image finding via `pointsrc.find_images()` with refinement
- Handles different source redshifts via `deflector.change_redshift()`

### Performance Optimizations
The code has been heavily optimized for speed (see `PERFORMANCE_OPTIMIZATIONS.md` for details):
- **Vectorized operations**: Replaced Python loops with NumPy broadcasting (50-100x faster)
- **Redshift grouping**: Groups sources by redshift to minimize expensive `change_redshift()` calls (10-30x faster)
- **Eliminated `.iterrows()`**: Uses direct array access instead of slow DataFrame iteration (50-100x faster)
- **Bulk I/O**: Writes CSV files in single operations instead of line-by-line
- **Overall speedup**: 4-8x faster for typical workflows with hundreds of sources

### Key Dependencies
- **lenstool** Python wrapper for deflection computations
- **pyLensLib**: deflector, pointsrc, srccatalog, lenstool utilities
- **astropy**: cosmology, table handling
- **scipy**: spatial operations (ConvexHull)
- **matplotlib**: visualization

## Typical Use Case

```bash
conda activate lenstool_env
python generate_lens_models.py cluster_model.par 100 \
  --fieldsize 100 --src_maglim 30 --img_maglim 28 \
  --lens --output_dir models/
```

Generates 100 cluster models with 100×100 arcsec² fields, detecting images brighter than mag=28.

## Output Files (per model)

### Always Generated
- `simulated_model_XXX.par`: Lenstool parameter file
- `simulated_model_XXX_sources.csv`: Source catalog
- `simulated_model_XXX_clmemb.csv`: Cluster member galaxies (mag < 22)

### Generated with `--lens` flag
- `simulated_model_XXX_allimages.csv`: All multiple images found
- `simulated_model_XXX_images.csv`: Images passing magnitude cut
- `simulated_model_XXX_with_images.png`: Convergence map with critical lines and multiple images overlaid

### Generated with `--test` flag
- `simulated_model_XXX.png`: Convergence map with critical lines

## Key Functions

### Measurement Functions
- `measureRadialDistribution()`: Bins galaxies by radius, returns surface density
- `fitNFWtoRadialDistribution()`: Fits projected NFW profile
- `measureLuminosityFunction()`: Bins galaxies by magnitude
- `fitSchechterFunction()`: Fits Schechter luminosity function
- `measureScalingRelations()`: Measures v_disp-mag and r_cut-mag relations

### Generation Functions
- `sampleMagnitudesFromFittedLF()`: Samples galaxy magnitudes from Schechter function
- `generateLenstoolModel()`: Creates randomized lens model with proper scaling relations
- `v_disp_from_mag()`: Assigns velocity dispersion from magnitude
- `cut_radius_from_mag()`: Assigns cut radius from magnitude

### I/O Functions
- `readLenstoolBlock()`: Parses Lenstool parameter blocks
- `writeLenstoolPar()`: Writes complete Lenstool parameter file
- `write_lenstool_image_catalog()`: Writes image catalog in Lenstool format

### Test/Visualization Functions
- `runTestRadialDistribution()`: Plots radial distribution + NFW fit
- `runTestLuminosityFunction()`: Plots luminosity function + Schechter fit
- `runTestScalingRelations()`: Plots scaling relations
- `runTestLenstoolModel()`: Visualizes convergence map with critical lines
- `plotMassMapWithImages()`: Creates convergence map with critical lines and multiple images overlaid (colored by source)

## Command-Line Arguments

### Required
- `parfile`: Input Lenstool .par file
- `nmodels`: Number of models to generate

### Optional (Galaxy Properties)
- `--mmin`, `--mmax`: Magnitude range for luminosity function (default: 17-26)
- `--scatter`: Lognormal scatter for scaling relations (default: 0.5)
- `--bcg`: Number of BCG galaxies to align with main potentials (default: 1)
- `--bcg_offset`, `--bcg_elloffset`, `--bcg_paoffset`: BCG positioning offsets

### Optional (Field Setup)
- `--fieldsize`: Side length of field of view in arcsec (default: 100)
- `--src_maglim`: Source magnitude limit (default: 30)
- `--img_maglim`: Image magnitude limit (default: 30)

### Optional (Behavior)
- `--lens`: Enable multiple image generation
- `--test`: Run validation tests and plots
- `--plotimages`: Plot multiple images on convergence map
- `--seed`: Random seed (default: 42)
- `--output_dir`: Output directory (default: "generated_models")

## Workflow Summary

1. Parse reference model from input `.par` file
2. Measure galaxy distribution properties (NFW, Schechter, scaling relations)
3. For each model iteration:
   - Sample new galaxy magnitudes from Schechter function
   - Assign positions (NFW), velocity dispersions, and cut radii
   - Optionally align BCGs with main dark matter halos
   - Write Lenstool parameter file
   - If `--lens`: Generate source catalog, ray-trace, find multiple images
   - If `--test`: Generate validation plots
4. Save all outputs to specified directory

## How the Number of Main Dark Matter Halos is Determined

The number of main DM halos is **determined from the reference model** and **preserved in the output**:

1. **From Reference Model**: The code reads the reference `.par` file and extracts main potentials using:
   ```python
   mainpot = selectPotentielByType(potentiel, ptype='main')
   ```
   This selects all potentials with `id` starting with 'O' (e.g., 'O1', 'O2', 'O3')

2. **Preserved in Output**: All main halos from the reference model are included in the generated models. The `generateLenstoolModel()` function randomizes their positions, ellipticities, and position angles while preserving the number of halos and their relative distances (within tolerance)

3. **BCG Association**: The `--bcg` parameter (default: 1) controls how many BCG galaxies are aligned with the main potentials. These BCG galaxies are positioned near the first N main halos (where N = `--bcg` value)

**Result**: Generated models have **the same number of main dark matter halos** as the reference model, with randomized properties but preserved structure.

## How the Number of Galaxies is Determined

The number of galaxies generated for each model is **NOT** a user-specified parameter. Instead, it is calculated automatically based on:

1. **Fitted Luminosity Function**: The code measures the luminosity function from the reference model and fits a Schechter function with parameters (φ*, M*, α)

2. **Magnitude Range**: User specifies `--mmin` and `--mmax` (default: 17-26)

3. **Field Area**: User specifies `--fieldsize` (default: 100 arcsec), which defines the area = (fieldsize/60)² arcmin²

4. **Integration**: The expected number of galaxies is computed as:
   ```
   N_expected = Area × ∫(mmin to mmax) φ* × 10^(0.4(α+1)(M*-m)) × exp(-10^(0.4(M*-m))) dm
   ```

5. **Sampling**: Exactly `N_expected` galaxy magnitudes are sampled from the fitted Schechter function using inverse transform sampling

**Example**: 
- Field size = 100 arcsec → Area = (100/60)² ≈ 2.78 arcmin²
- Fitted Schechter: φ* = 8.3×10⁻³, M* = 18.5, α = -1.15
- Magnitude range: 17-26
- Result: ~300-500 galaxies (actual number depends on the integral)

**To control the number of galaxies**, adjust:
- `--mmin`, `--mmax`: Narrower range → fewer galaxies
- `--fieldsize`: Smaller field → fewer galaxies (scales as area²)
- The reference model's luminosity function (φ*) determines the base density

