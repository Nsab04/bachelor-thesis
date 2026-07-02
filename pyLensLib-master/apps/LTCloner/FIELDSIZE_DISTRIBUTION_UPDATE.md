# Final Configuration: Galaxy Distribution with Fieldsize

## What Was Changed

The code has been updated to distribute galaxies within `rmax = fieldsize/2 × √2` (the full extent based on fieldsize), while maintaining correct number density from the luminosity function.

## Current Behavior

### Galaxy Distribution:
- **rmax (placement radius)**: `fieldsize/2 × √2` 
  - For fieldsize=400: rmax = 282.8 arcsec
  - Galaxies are distributed within this full radius
  
### Number of Galaxies:
- **Sampling**: Based on area = π × rmax²
- **Density**: Maintained from input luminosity function
- **Result**: More galaxies for larger fieldsize, same density

## How It Works

1. **Measure input density** from the original .par file:
   ```
   number_density = N_input / area_input
   ```

2. **Calculate rmax from fieldsize**:
   ```
   rmax = fieldsize/2 × √2
   area_sampling = π × rmax²
   ```

3. **Sample galaxies to maintain density**:
   ```
   N_galaxies = number_density × area_sampling
   ```

4. **Distribute within rmax**:
   - Galaxies placed using NFW profile within rmax
   - Larger fieldsize → larger rmax → more galaxies
   - But density (galaxies per unit area) stays constant

## Example with Your Config

From `ltcloner_config.yaml`:
```yaml
fieldsize: 400.0  # arcsec
```

Results:
- **rmax**: 400/2 × √2 = 282.8 arcsec
- **Area**: π × 282.8² / 3600 = 69.6 arcmin²
- **If input density is 100 gal/arcmin²**: N ≈ 6960 galaxies
- All 6960 galaxies distributed within r < 282.8"

## Comparison: Different Fieldsizes

| Fieldsize | rmax | Area | N (if density=100) | Density |
|-----------|------|------|-------------------|---------|
| 200" | 141.4" | 17.4 arcmin² | ~1740 gal | 100 gal/arcmin² |
| 400" | 282.8" | 69.6 arcmin² | ~6960 gal | 100 gal/arcmin² |
| 600" | 424.3" | 156.6 arcmin² | ~15660 gal | 100 gal/arcmin² |

✅ **Density constant, N scales with fieldsize², rmax scales with fieldsize**

## Key Points

1. **Luminosity function sampling is correct**: 
   - Uses the measured number density from input
   - Scales appropriately with area

2. **Distribution extends to full fieldsize**:
   - Galaxies can be placed anywhere within rmax = fieldsize/2 × √2
   - Not limited to original cluster extent

3. **Number density is maintained**:
   - Same galaxies per unit area
   - But total number increases for larger fieldsize (covering more area)

## Console Output Example

When you run ltcloner with fieldsize=400, you'll see:
```
Input galaxy distribution:
  Area: 17.45 arcmin² (convex hull)
  N galaxies: 1745
  Number density: 100.0 galaxies/arcmin²

Output galaxy placement:
  rmax: 282.8 arcsec (from fieldsize=400.0)
  Effective area: 69.60 arcmin²
  Sampling area: 69.60 arcmin² (based on fieldsize)
  Expected N galaxies: 6960
  → Maintaining input number density: 100.0 gal/arcmin²
  → More galaxies for larger fieldsize, but same density
```

## Files Modified

- **apps/ltcloner.py** (lines ~583-604): Reverted to using fieldsize for rmax calculation

## Status

✅ **Implemented** - Galaxies now distributed within fieldsize/2 extent
✅ **Density conserved** - Number density maintained from luminosity function
✅ **Ready to use** - Your config file with fieldsize=400 will work correctly

---

**Summary**: Galaxies are now distributed throughout the full `fieldsize/2 × √2` radius, with the number of galaxies scaling to maintain constant number density from the input luminosity function. This is the correct behavior for exploring different field sizes while maintaining realistic galaxy distributions.

