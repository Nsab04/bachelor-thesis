# Quick Start: Using ltcloner.py with YAML Configuration

## Step 1: Install Dependencies

Make sure you have PyYAML installed:
```bash
pip install pyyaml
```

## Step 2: Create Your Configuration File

Copy and edit the example configuration:
```bash
cp ltcloner_config.yaml my_config.yaml
```

Edit `my_config.yaml` with your parameters:
```yaml
# Minimal example - only required parameters
parfile: "path/to/your/cluster.par"
nmodels: 10
```

## Step 3: Run ltcloner

```bash
python ltcloner.py my_config.yaml
```

That's it! The script will:
1. Read your configuration
2. Print the loaded parameters
3. Process the models
4. Save output to the specified directory

## Common Use Cases

### Quick Test Run
```yaml
parfile: "test_cluster.par"
nmodels: 1
test: true
output_dir: "test_run"
```

Run: `python ltcloner.py test_config.yaml`

### Production Run with Lensing
```yaml
parfile: "final_cluster.par"
nmodels: 100
lens: true
compute_cs: true
output_dir: "production_models"
seed: 12345
```

Run: `python ltcloner.py production_config.yaml`

### Elongated Cluster (60° opening angle)
```yaml
parfile: "elliptical_cluster.par"
nmodels: 50
opening_angle: 60.0
scatter: 0.3
bcg: 3
output_dir: "elongated_models"
```

Run: `python ltcloner.py elongated_config.yaml`

## Troubleshooting

**Problem**: `FileNotFoundError: Configuration file not found`
**Solution**: Check the path to your YAML file

**Problem**: `KeyError: 'parfile'`
**Solution**: Make sure your config file has the required `parfile` and `nmodels` parameters

**Problem**: `ImportError: No module named 'yaml'`
**Solution**: Install PyYAML: `pip install pyyaml`

## Getting Help

View all parameters and their defaults:
```bash
cat ltcloner_config.yaml
```

Read the complete guide:
```bash
cat LTCLONER_YAML_GUIDE.md
```

Test your setup:
```bash
python test_ltcloner_yaml.py
```

## Example Output

When you run ltcloner with a config file, you'll see:
```
Configuration:
  parfile: cluster.par
  nmodels: 10
  output_dir: generated_models
  fieldsize: 100.0 arcsec
  seed: 42

======================================================================
Estimated area: 123.45 arcmin² from convex hull
------------------
Number of galaxies found: 150
Measured scaling relations: alpha = 0.23, beta = 0.64
Fitted NFW parameters: n0 = 1.30e-02, rs = 150.00 arcsec
...
```

## Pro Tips

1. **Use descriptive config names**: `high_scatter_test.yaml` instead of `config1.yaml`
2. **Add comments**: Document why you chose specific parameters
3. **Version control**: Track your configs in git
4. **Keep a library**: Maintain configs for different cluster types
5. **Test first**: Always do a test run with `nmodels: 1` and `test: true`

## Next Steps

- Customize parameters in your config file
- Run multiple configurations for comparison
- Share configs with your team
- Track configurations in version control

