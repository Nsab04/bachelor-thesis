#!/usr/bin/env python3
"""
Test script to verify YAML configuration loading for ltcloner.py
"""

import sys
import yaml
import os

class Config:
    """Configuration class to load parameters from YAML file (copy for testing)"""

    def __init__(self, config_file):
        """Load configuration from YAML file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Required parameters
        self.parfile = config['parfile']
        self.nmodels = config['nmodels']

        # Magnitude limits
        self.mmin = config.get('mmin', 17.0)
        self.mmax = config.get('mmax', 26.0)

        # Random seed
        self.seed = config.get('seed', 42)

        # Distribution measurement
        self.nbinr = config.get('nbinr', 5)

        # Field of view
        self.fieldsize = config.get('fieldsize', 100.0)

        # Source and image parameters
        self.src_maglim = config.get('src_maglim', 30.0)
        self.img_maglim = config.get('img_maglim', 30.0)

        # BCG parameters
        self.bcg = config.get('bcg', 1)
        self.bcg_offset = config.get('bcg_offset', 1.0)
        self.bcg_elloffset = config.get('bcg_elloffset', 0.2)
        self.bcg_paoffset = config.get('bcg_paoffset', 10.0)

        # Model geometry
        self.opening_angle = config.get('opening_angle', 180.0)

        # Scaling relations
        self.scatter = config.get('scatter', 0.5)
        self.tolerance = config.get('tolerance', 0.1)

        # Flags
        self.nullify_ellipticity = config.get('nullify_ellipticity', False)
        self.lens = config.get('lens', False)
        self.test = config.get('test', False)
        self.compute_cs = config.get('compute_cs', False)

        # Output
        self.output_dir = config.get('output_dir', 'generated_models')

    def __repr__(self):
        """String representation of configuration"""
        lines = ["Configuration:"]
        lines.append(f"  parfile: {self.parfile}")
        lines.append(f"  nmodels: {self.nmodels}")
        lines.append(f"  output_dir: {self.output_dir}")
        lines.append(f"  fieldsize: {self.fieldsize} arcsec")
        lines.append(f"  seed: {self.seed}")
        return "\n".join(lines)

def test_config_loading():
    """Test loading configuration from YAML file"""
    print("Testing YAML configuration loading...\n")

    # Create a test config file
    test_config = """
# Test configuration
parfile: "test_input.par"
nmodels: 5
mmin: 18.0
mmax: 24.0
seed: 999
fieldsize: 120.0
opening_angle: 90.0
scatter: 0.3
lens: true
test: true
output_dir: "test_models"
"""

    test_file = "test_ltcloner_config.yaml"

    try:
        # Write test config
        with open(test_file, 'w') as f:
            f.write(test_config)

        # Load config
        config = Config(test_file)

        # Verify parameters
        print("✓ Config loaded successfully\n")
        print(config)
        print("\n" + "="*60)

        # Test individual parameters
        tests = [
            (config.parfile == "test_input.par", "parfile"),
            (config.nmodels == 5, "nmodels"),
            (config.mmin == 18.0, "mmin"),
            (config.mmax == 24.0, "mmax"),
            (config.seed == 999, "seed"),
            (config.fieldsize == 120.0, "fieldsize"),
            (config.opening_angle == 90.0, "opening_angle"),
            (config.scatter == 0.3, "scatter"),
            (config.lens == True, "lens flag"),
            (config.test == True, "test flag"),
            (config.output_dir == "test_models", "output_dir"),
        ]

        print("\nParameter verification:")
        all_passed = True
        for passed, name in tests:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n✓ All tests PASSED!")
            return 0
        else:
            print("\n✗ Some tests FAILED!")
            return 1

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nCleaned up test file: {test_file}")

def test_default_values():
    """Test that default values are applied correctly"""
    print("\n" + "="*60)
    print("Testing default values...\n")

    # Minimal config with only required parameters
    minimal_config = """
parfile: "minimal.par"
nmodels: 1
"""

    test_file = "minimal_config.yaml"

    try:
        with open(test_file, 'w') as f:
            f.write(minimal_config)

        config = Config(test_file)

        print("✓ Loaded minimal config\n")

        # Check defaults
        tests = [
            (config.mmin == 17.0, "mmin default"),
            (config.mmax == 26.0, "mmax default"),
            (config.seed == 42, "seed default"),
            (config.fieldsize == 100.0, "fieldsize default"),
            (config.opening_angle == 180.0, "opening_angle default"),
            (config.scatter == 0.5, "scatter default"),
            (config.tolerance == 0.1, "tolerance default"),
            (config.lens == False, "lens default"),
            (config.test == False, "test default"),
            (config.output_dir == "generated_models", "output_dir default"),
        ]

        print("Default value verification:")
        all_passed = True
        for passed, name in tests:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n✓ All default value tests PASSED!")
            return 0
        else:
            print("\n✗ Some default value tests FAILED!")
            return 1

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    result1 = test_config_loading()
    result2 = test_default_values()

    print("\n" + "="*60)
    if result1 == 0 and result2 == 0:
        print("✓ ALL TESTS PASSED!")
        print("\nYAML configuration system is working correctly.")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED!")
        sys.exit(1)

