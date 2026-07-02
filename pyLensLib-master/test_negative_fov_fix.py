#!/usr/bin/env python3
"""Test that getFoV correctly reads negative FOV values"""

from pyLensLib.lenstool import getFoV, isfloat
import numpy as np
import os

print("=" * 60)
print("Testing isfloat function with negative numbers")
print("=" * 60)

test_values = [
    ('123', True),
    ('-123', True),
    ('1.5', True),
    ('-1.5', True),
    ('-50.0', True),
    ('50.0', True),
    ('1e-5', True),
    ('-1e-5', True),
    ('1.23e10', True),
    ('-1.23e10', True),
    ('+123.45', True),
    ('abc', False),
    ('-', False),
    ('', False),
]

all_passed = True
for val, expected in test_values:
    result = isfloat(val)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"  {status} isfloat('{val}') = {result} (expected {expected})")

print(f"\nisfloat tests: {'PASSED' if all_passed else 'FAILED'}")

print("\n" + "=" * 60)
print("Testing getFoV with negative FOV values")
print("=" * 60)

# Create test .par files with different FOV configurations
test_cases = [
    ("Centered FOV (-50 to 50)", """
runmode
    reference 3 0 0
    end
champ
    xmin -50.0
    xmax 50.0
    ymin -50.0
    ymax 50.0
    end
""", [-50.0, 50.0, -50.0, 50.0]),

    ("Positive offset FOV (0 to 100)", """
runmode
    reference 3 0 0
    end
champ
    xmin 0.0
    xmax 100.0
    ymin 0.0
    ymax 100.0
    end
""", [0.0, 100.0, 0.0, 100.0]),

    ("Negative offset FOV (-100 to 0)", """
runmode
    reference 3 0 0
    end
champ
    xmin -100.0
    xmax 0.0
    ymin -100.0
    ymax 0.0
    end
""", [-100.0, 0.0, -100.0, 0.0]),

    ("Mixed FOV (-25 to 75)", """
runmode
    reference 3 0 0
    end
champ
    xmin -25.0
    xmax 75.0
    ymin -25.0
    ymax 75.0
    end
""", [-25.0, 75.0, -25.0, 75.0]),
]

all_tests_passed = True
for test_name, par_content, expected_lims in test_cases:
    test_file = f'test_fov_{test_name.replace(" ", "_").replace("(", "").replace(")", "")}.par'

    # Write test file
    with open(test_file, 'w') as f:
        f.write(par_content)

    # Test getFoV
    lims = getFoV(test_file)

    # Check results
    passed = np.allclose(lims, expected_lims)
    status = "✓" if passed else "✗"

    print(f"\n{status} Test: {test_name}")
    print(f"  Expected: xmin={expected_lims[0]}, xmax={expected_lims[1]}, ymin={expected_lims[2]}, ymax={expected_lims[3]}")
    print(f"  Got:      xmin={lims[0]}, xmax={lims[1]}, ymin={lims[2]}, ymax={lims[3]}")

    if not passed:
        all_tests_passed = False
        print(f"  ERROR: Values don't match!")

    # Cleanup
    os.remove(test_file)

print("\n" + "=" * 60)
if all_passed and all_tests_passed:
    print("✓ ALL TESTS PASSED!")
    print("getFoV now correctly handles negative FOV values.")
else:
    print("✗ SOME TESTS FAILED")
print("=" * 60)

