# ltcloner.py - YAML Configuration Guide

## Overview

The `ltcloner.py` script has been updated to use YAML configuration files instead of command-line arguments. This makes it easier to:
- Manage complex configurations
- Reuse parameter sets
- Document parameter choices
- Version control configurations

## Migration from Command Line to YAML

### Old Usage (Command Line)
```bash
python ltcloner.py input.par 10 \
    --mmin 17.0 \
    --mmax 26.0 \
    --seed 42 \
    --fieldsize 100.0 \
    --opening_angle 90.0 \
    --scatter 0.5 \
    --output_dir my_models \
    --lens
```

### New Usage (YAML)
```bash
python ltcloner.py config.yaml
```

## Configuration File Format

Create a YAML file (e.g., `ltcloner_config.yaml`) with your parameters:

```yaml
# Required parameters
parfile: "path/to/input.par"
nmodels: 10

# Optional parameters with defaults
mmin: 17.0
mmax: 26.0
seed: 42
fieldsize: 100.0
opening_angle: 180.0
scatter: 0.5
output_dir: "generated_models"

# Boolean flags
lens: false
test: false
nullify_ellipticity: false
compute_cs: false
```

## Complete Parameter Reference

### Required Parameters

- **parfile** (string): Path to input Lenstool .par file
- **nmodels** (int): Number of lens models to generate

### Magnitude Parameters

- **mmin** (float, default: 17.0): Minimum magnitude for luminosity function
- **mmax** (float, default: 26.0): Maximum magnitude for luminosity function

### Random Seed

- **seed** (int, default: 42): Random seed for reproducibility

### Field of View

- **fieldsize** (float, default: 100.0): Side length of the field of view (arcsec)

### Source and Image Parameters

- **src_maglim** (float, default: 30.0): Magnitude limit for source galaxies (intrinsic)
- **img_maglim** (float, default: 30.0): Magnitude limit for images (lensed sources)

### BCG (Brightest Cluster Galaxy) Parameters

- **bcg** (int, default: 1): Number of BCG galaxies to associate with main potentials
- **bcg_offset** (float, default: 1.0): Maximum offset for BCG galaxies from main potential center (arcsec)
- **bcg_elloffset** (float, default: 0.2): Maximum offset for BCG galaxy ellipticity
- **bcg_paoffset** (float, default: 10.0): Maximum offset for BCG galaxy position angle (degrees)

### Model Geometry

- **opening_angle** (float, default: 180.0): Force alignment of mainpot potentials (degrees)
  - 180.0 = full circle distribution
  - Smaller values create elongated, bimodal distributions

### Scaling Relations

- **scatter** (float, default: 0.5): Lognormal scatter for scaling relations
- **tolerance** (float, default: 0.1): Tolerance for distances between main potentials (fraction)

### Distribution Measurement

- **nbinr** (int, default: 5): Number of radial bins for distribution measurement

### Boolean Flags

- **nullify_ellipticity** (bool, default: false): Set ellipticity of all galaxies to zero
- **lens** (bool, default: false): Run with generation of multiple images for each model
- **test** (bool, default: false): Run in test mode without saving models
- **compute_cs** (bool, default: false): Compute cross sections and Einstein radii for each model

### Output

- **output_dir** (string, default: "generated_models"): Directory to save output models

## Example Configurations

### 1. Quick Test Run

```yaml
# test_config.yaml
parfile: "cluster.par"
nmodels: 1
test: true
output_dir: "test_output"
```

Usage: `python ltcloner.py test_config.yaml`

### 2. Production Run with Lensing

```yaml
# production_config.yaml
parfile: "cluster_final.par"
nmodels: 100
seed: 12345
mmin: 18.0
mmax: 25.0
fieldsize: 150.0
opening_angle: 90.0
scatter: 0.3
lens: true
compute_cs: true
output_dir: "production_models"
```

Usage: `python ltcloner.py production_config.yaml`

### 3. Elliptical Cluster with BCGs

```yaml
# elliptical_cluster.yaml
parfile: "elliptical_cluster.par"
nmodels: 50
opening_angle: 60.0
bcg: 3
bcg_offset: 0.5
bcg_elloffset: 0.1
bcg_paoffset: 5.0
scatter: 0.4
tolerance: 0.15
output_dir: "elliptical_models"
```

Usage: `python ltcloner.py elliptical_cluster.yaml`

## Error Handling

The script will provide clear error messages if:
- Configuration file is not found
- YAML syntax is invalid
- Required parameters are missing
- Parameter values are invalid

Example error:
```
Error: Configuration file not found: config.yaml
```

## Benefits of YAML Configuration

1. **Readability**: Easy to read and understand parameter values
2. **Comments**: Document why you chose specific values
3. **Version Control**: Track configuration changes in git
4. **Reproducibility**: Share exact configurations with collaborators
5. **Multiple Configs**: Maintain different configurations for different use cases
6. **No Shell Escaping**: No issues with spaces or special characters in paths

## Backwards Compatibility

The old command-line interface is no longer supported. To migrate:

1. Create a YAML file with your parameters
2. Run with: `python ltcloner.py your_config.yaml`

## Tips

- Use comments (`#`) to document your configuration choices
- Keep different YAML files for different experiments
- Use meaningful names for your config files (e.g., `high_scatter.yaml`, `test_run.yaml`)
- You can omit parameters to use their default values
- YAML is whitespace-sensitive - use spaces, not tabs

## Dependencies

The script now requires the `pyyaml` package. Install with:
```bash
pip install pyyaml
```

Or if using conda:
```bash
conda install pyyaml
```

