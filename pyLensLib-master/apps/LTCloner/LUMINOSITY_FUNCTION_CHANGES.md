# Summary: Changes to Luminosity Function Sampling in ltcloner.py

## What Changed

### 1. Area Calculation (Line ~552) - **FIXED BUG**

**Before (WRONG)**:
```python
area_out_arcmin2 = (args.fieldsize / 60.0) ** 2  # INCORRECT!
```
This calculated `(fieldsize/60)²` which gave wrong units and wrong value.

**After (CORRECT)**:
```python
# Calculate output area: circular area within rmax = fieldsize/2 × sqrt(2)
rmax_output = args.fieldsize / 2.0 * np.sqrt(2.0)
area_out_arcmin2 = np.pi * rmax_output**2 / 3600.0  # Convert arcsec² to arcmin²
```

**Why this is correct**:
- Galaxies are distributed within `rmax = fieldsize/2 × √2` (diagonal of square FOV)
- Circular area: `π × rmax²`
- Convert from arcsec² to arcmin²: divide by 3600

**Example**:
- fieldsize = 400 arcsec
- rmax = 400/2 × 1.414 = 282.8 arcsec
- area_out_arcmin2 = π × 282.8² / 3600 = **69.6 arcmin²** ✓
- Old (wrong): (400/60)² = 44.4 arcmin² ✗

### 2. Luminosity Function Sampling (Line ~649) - **INTENTIONAL CHANGE**

**Before (during revert)**:
```python
area=area_arcmin2  # Input area only
```

**After (CURRENT)**:
```python
area=area_out_arcmin2  # Output area based on fieldsize
```

**Why this change**:
- You want more galaxies for larger fieldsize
- `area_out_arcmin2` scales with fieldsize
- This maintains constant **number density** (galaxies per unit area)

### 3. Correction Factor (Lines 651-654) - **EXISTING CODE**

The code also samples with input area for comparison:
```python
mags_0 = sampleMagnitudesFromFittedLF(..., area=area_arcmin2, ...)
f_corr = len(gals)/len(mags_0)  # Correction factor for density mismatch
```

This correction factor accounts for any density differences between input and sampled distributions.

## How It Works Now

### Complete Flow:

1. **Fit luminosity function to INPUT data**:
   - Uses `area_arcmin2` (from input convex hull)
   - Gets Schechter parameters: `phistar`, `Mstar`, `alpha_m`

2. **Calculate OUTPUT area from fieldsize**:
   - `rmax = fieldsize/2 × √2`
   - `area_out_arcmin2 = π × rmax² / 3600`

3. **Sample magnitudes for OUTPUT**:
   - `sampleMagnitudesFromFittedLF(..., area=area_out_arcmin2, ...)`
   - Number of galaxies ∝ `area_out_arcmin2`
   - **More galaxies for larger fieldsize!**

4. **Distribute galaxies spatially**:
   - Use NFW with observed `rs` (fixed)
   - Within `rmax = fieldsize/2 × √2`
   - **Number density n(r) stays constant!** (due to NFW sampling fix)

## Expected Behavior

| Fieldsize | rmax | Area | N galaxies (if density=100) |
|-----------|------|------|---------------------------|
| 100" | 70.7" | 4.4 arcmin² | ~440 |
| 200" | 141.4" | 17.4 arcmin² | ~1740 |
| 400" | 282.8" | 69.6 arcmin² | ~6960 |

**Key point**: N scales with area (fieldsize²), maintaining constant density!

## What This Achieves

✅ **Luminosity function correctly samples based on output area**
✅ **More galaxies for larger fieldsize** (N ∝ fieldsize²)
✅ **Number density maintained** (galaxies per arcmin² constant)
✅ **Observed rs preserved** (from NFW fit to input data)

## Summary

### Changes Made:
1. **Fixed bug**: area_out_arcmin2 calculation (was wrong unit conversion)
2. **Intentional**: Use area_out_arcmin2 for LF sampling (scales with fieldsize)
3. **NFW fix**: Sample from [0, CDF(rmax)] to maintain constant n(r)

### Result:
The code now correctly:
- Samples more galaxies for larger fieldsize
- Maintains constant number density at all radii
- Preserves the observed NFW scale radius

All changes work together to achieve your goal: **constant number density as a function of radius, regardless of fieldsize!**

