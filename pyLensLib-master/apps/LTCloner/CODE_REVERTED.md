# Code Reverted to Original State

## Date: January 23, 2026

## Summary

All changes related to number density conservation have been **reverted**. The code is now back to its original state.

## Files Reverted

### 1. pyLensLib/lenstool.py
**Function**: `sample_radius_from_projected_nfw`

**Reverted to**: Original simple implementation
```python
def sample_radius_from_projected_nfw(rs, size, rmax=5.0, rmin=0.05, ngrid=5000, seed=None):
    rng = np.random.default_rng(seed)
    r_vals = np.logspace(np.log10(rmin), np.log10(rmax), ngrid)
    x_vals = r_vals / rs
    cdf_vals = _f_proj_nfw(x_vals)
    cdf_vals /= cdf_vals[-1]  # Normalize to 1
    u = rng.uniform(0, 1, size)
    r_samples = np.interp(u, cdf_vals, r_vals)
    return r_samples
```

**Removed**: All attempts to fix CDF normalization for density conservation

### 2. apps/ltcloner.py

**Reverted changes**:
1. ❌ Removed number density tracking code (lines ~582-610)
2. ❌ Removed rmax calculation based on fieldsize for placement
3. ❌ Removed area_for_sampling variable
4. ❌ Removed rs scaling with fieldsize
5. ❌ Removed all diagnostic print statements about density

**Back to original**:
- Simple area calculation from convex hull
- Direct use of `args.fieldsize/2.0 * np.sqrt(2.0)` for rmax in generateLenstoolModel
- Original `rs*args.rs_fact` without scaling
- Original `area_arcmin2` for luminosity function sampling

## Current State

The code now works exactly as it did before all the number density conservation attempts:

```python
# Original behavior (restored):
- rmax = args.fieldsize/2.0 * √2  (scales with fieldsize)
- N_galaxies sampled from LF based on area_arcmin2 (from input convex hull)
- rs = rs_fitted * rs_fact (no fieldsize scaling)
- Galaxies distributed via NFW(rs) within rmax
```

## Known Behavior (Not Changed)

As documented in the analysis, the current implementation:
- ✅ Samples from luminosity function correctly
- ✅ Distributes galaxies within fieldsize/2 extent
- ✅ Uses NFW profile for galaxy placement
- ❌ Does **NOT** conserve number density n(r) when fieldsize changes

This is the **original behavior** and is **mathematically expected** given the NFW sampling approach.

## Validation

- ✅ Code compiles without errors
- ✅ Only pre-existing warnings remain
- ✅ All experimental changes removed
- ✅ Original functionality restored

## Documentation Created (Preserved)

The following analysis documents remain for future reference:
- `NFW_FUNDAMENTAL_PROBLEM.md` - Why density conservation is mathematically impossible with current approach
- `FINAL_DENSITY_ANALYSIS.md` - Complete analysis of all attempted solutions
- `test_density_conservation.py` - Test script (can be used to verify behavior)

## Next Steps (If Needed)

If you want to conserve number density n(r) in the future, the only working approach is:
- **Constant cluster**: Keep rmax, N, and rs constant (independent of fieldsize)
- This makes fieldsize a pure visualization parameter

The current reverted code maintains the original flexibility of distributing galaxies throughout the fieldsize extent.

---

**Status**: ✅ **REVERTED TO ORIGINAL**
**Date**: January 23, 2026
**Validation**: Passed - code compiles and runs

