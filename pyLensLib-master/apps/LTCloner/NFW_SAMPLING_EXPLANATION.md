# Understanding the NFW Sampling and Number Density Issue

## The Core Problem

You correctly identified that when `fieldsize` increases:
1. More galaxies are sampled (from luminosity function) ✓
2. They're distributed via NFW within rmax = fieldsize/2 × √2 ✓
3. But the inner density changes ✗

## Why This Happens - The Mathematics

### Scenario: Fixed N, Varying rmax

When you sample **N galaxies** from NFW and truncate at different `rmax`:

```
Truncated at rmax=70": 
- CDF normalized to f(70/rs)
- Probability of r<50 given r<70 = f(50/rs) / f(70/rs) ≈ 0.722

Truncated at rmax=200":
- CDF normalized to f(200/rs)  
- Probability of r<50 given r<200 = f(50/rs) / f(200/rs) ≈ 0.359
```

**Result**: With larger rmax, fewer galaxies land in inner regions (as fraction of total).

This is **mathematically correct** for truncated NFW! The problem is that you want **constant density**, not a truncated NFW.

## The Solution

### Option 1: Don't Truncate at rmax (RECOMMENDED)

Sample from the **full, untruncated NFW** and just place galaxies up to rmax:

```python
# Sample from untruncated NFW (use large r_max_infinity)
r_max_infinity = 1000 * rs  # Effectively infinity
r_samples = sample_from_untruncated_nfw(rs, size, r_max_infinity)

# Only use galaxies within rmax
r_samples_truncated = r_samples[r_samples <= rmax]
```

But this means you don't get exactly `size` galaxies - you get fewer!

### Option 2: Scale the NFW Scale Radius

Scale `rs` proportionally with `fieldsize`:

```python
rs_scaled = rs_fitted * (fieldsize / fieldsize_original)
```

This keeps the **relative** density profile the same.

### Option 3: Accept the Current Behavior

The current code is **mathematically correct** for what it does:
- Sample N galaxies from truncated NFW(rs, rmax)
- The truncation naturally causes density changes

## What Your Code Should Do

Based on your description, here's what I think you want:

**When fieldsize increases:**
1. Sample more galaxies (N ∝ fieldsize²) ✓ Already works
2. The NFW profile should "scale" with the cluster size
3. Inner regions maintain the same density

**This requires**: `rs` should scale with `rmax` (fieldsize)!

## The Correct Fix

In your code where you call `sample_radius_from_projected_nfw`:

```python
# Calculate rs that scales with fieldsize
rs_scaled = rs_fitted * (rmax / rmax_fitted)

# Now sample with scaled rs
r_samples = sample_radius_from_projected_nfw(
    rs=rs_scaled,  # Scaled, not fixed!
    size=n_galaxies,
    rmax=rmax
)
```

This way:
- x = r/rs stays in the same range
- The relative density profile is preserved
- Absolute density stays constant

## Summary

The NFW sampling function is **correct as-is**. The issue is that you need to **scale rs with fieldsize** when calling it, not change the sampling function itself.

The fix belongs in the **calling code** (ltcloner.py or generateLenstoolModel), not in sample_radius_from_projected_nfw().

