# ❌ TEST RESULT: Number Density NOT Conserved

## Date: January 23, 2026

## Test Performed

Checked if the current implementation in ltcloner.py conserves the number density n(r) as a function of radius when fieldsize is changed.

## Result: ❌ **FAILED**

The number density is **NOT conserved**. It increases significantly with fieldsize.

## Test Data

### Density in Common Region (r < 60")

| Fieldsize | rmax | N_total | N(r<60") | Density (gal/arcmin²) | Change |
|-----------|------|---------|----------|----------------------|--------|
| 100" | 70.7" | 436 | 348 | **110.8** | baseline |
| 150" | 106.1" | 981 | 785 | **249.9** | 2.3× |
| 200" | 141.4" | 1745 | 1386 | **441.2** | 4.0× |
| 250" | 176.8" | 2727 | 2181 | **694.2** | 6.3× |

### Radial Profile Comparison

At radius r=27":
- FS=100: 106.1 gal/arcmin²
- FS=150: 314.8 gal/arcmin² (3.0×)
- FS=200: 626.0 gal/arcmin² (5.9×)
- FS=250: 1018.6 gal/arcmin² (9.6×)

**Average difference: 539.88%**
**Maximum difference: 860.00%**

## The Problem

The current implementation:

```python
# Line ~593: rmax scales with fieldsize
rmax_for_placement = args.fieldsize / 2.0 * np.sqrt(2.0)

# Line ~605: Sample more galaxies for larger area
area_for_sampling = π × rmax²
N_galaxies = density × area_for_sampling
```

This causes:
1. **rmax increases** with fieldsize (linearly)
2. **Area increases** as rmax² (quadratically)
3. **N galaxies sampled** increases with area (quadratically)
4. But galaxies are **sampled from the same NFW profile shape**

The problem is that when you sample more galaxies (say 4× more) and distribute them according to the SAME NFW(rs=30") profile, the **inner regions get more galaxies too**!

## Why This Happens

### NFW Profile Behavior

The NFW profile is: n(r) ∝ 1/(r/rs × (1+r/rs)²)

When you:
1. Sample 4× more galaxies (because area is 4× larger)
2. All from NFW with same rs
3. The sampling puts galaxies at ALL radii according to the NFW shape

Result: Inner bins (r < 60") get roughly 4× more galaxies too, but their area hasn't changed, so **density increases**!

## What You Originally Asked For

You wanted: **"The number density as a function of distance should remain constant"**

This means:
- n(r=10") should be the same for fieldsize=100 and fieldsize=200
- n(r=30") should be the same for all fieldsizes
- The radial profile n(r) should overlay perfectly

## Current Status

❌ **NOT ACHIEVED** - Density increases with fieldsize at all radii

## What This Means

The current code is sampling the luminosity function correctly (more galaxies for larger area), but the spatial distribution is NOT maintaining constant number density. The problem is fundamental to how NFW sampling works when you scale the total number.

## Possible Solutions

### Option 1: Keep rmax constant (previous approach)
- rmax from original cluster extent
- Same N galaxies regardless of fieldsize
- **Pros**: n(r) perfectly conserved
- **Cons**: Doesn't fill larger fieldsize

### Option 2: Use different rs for different fieldsizes
- Scale rs with fieldsize: rs_new = rs_original × (fieldsize_new/fieldsize_original)
- **Pros**: Could maintain density
- **Cons**: Changes cluster structure, complex

### Option 3: Sample different density at different radii
- More complex: sample galaxies in annuli, maintaining density in each
- **Pros**: Could work
- **Cons**: Very complex to implement

## Recommendation

The current implementation **does not** conserve number density n(r). You need to decide:

1. **Keep cluster properties constant** (Option 1: constant rmax)
   - Same cluster, different FOV views
   - n(r) perfectly conserved
   - Fieldsize is just visualization parameter

2. **Allow cluster to "grow" with fieldsize** (current approach)
   - More galaxies, larger extent
   - But density INCREASES (not conserved)
   - Current behavior

3. **Scale cluster properly** (Option 2 or 3)
   - Complex implementation needed
   - Would conserve density but change other properties

## Test Script

The test is available at:
```
/Users/maxmen3/projects/pyLensLib/apps/test_density_conservation.py
```

Run with:
```bash
python3 test_density_conservation.py
```

## Conclusion

❌ **The current implementation does NOT conserve number density n(r) when fieldsize changes.**

The density increases by factors of 2-9× depending on fieldsize, which is **not** the desired behavior based on your original requirement.

