# The Fundamental Problem with NFW Sampling and Fieldsize

## The Issue

You cannot maintain **both** of these simultaneously:
1. Sample more galaxies for larger fieldsize (N ∝ fieldsize²)
2. Distribute them according to a fixed NFW(rs) profile
3. Keep number density n(r) constant at all radii

**This is mathematically impossible with a fixed NFW profile.**

## Why It's Impossible

### The NFW Profile
NFW has a characteristic scale radius `rs`. The shape of n(r) is determined by rs:
- n(r) ~ 1/(r/rs) × (1 + r/rs)²  for projected NFW

### What Happens When You Scale N
If you sample 4× more galaxies from NFW(rs=30"):
- At r=10": you get ~4× more galaxies
- At r=50": you get ~4× more galaxies  
- At r=100": you get ~4× more galaxies

But:
- The **area** of the r=10" annulus hasn't changed
- So **density** at r=10" increases by 4×

### The Core Problem
When rmax increases from 70" to 280" (4× in radius):
- Total area increases 16×
- You sample 16× more galaxies
- All distributed via same NFW(rs=30")
- Inner regions (r<70") get way more galaxies
- But their area is the same
- **Density must increase!**

## The Solutions

### Solution 1: Scale rs with fieldsize ✓
```python
rs_scaled = rs_original × (fieldsize / fieldsize_original)
```

**Logic**:
- Larger fieldsize → larger cluster scale
- NFW shape adapts to cluster size
- n(r_physical) stays constant

**Implementation**:
```python
# In ltcloner.py, line ~620
rs_for_sampling = rs * (args.fieldsize / original_fieldsize)
model_smooth, model_gal, model_gas = generateLenstoolModel(
    ..., rs=rs_for_sampling, rmax=rmax_for_placement, ...
)
```

### Solution 2: Use uniform distribution (not NFW) ✓
```python
# Sample uniformly in area
theta = rng.uniform(0, 2π, N)
r = rmax × sqrt(rng.uniform(0, 1, N))
```

**Logic**:
- Galaxies distributed uniformly in area
- n(r) = constant by construction
- No profile shape issues

### Solution 3: Keep cluster constant (original approach) ✓
```python
rmax = constant (from original input)
N = constant
n(r) = perfectly constant
```

**Logic**:
- Same cluster, different FOV views
- Most physically correct
- Fieldsize is visualization parameter only

## Recommendation

Based on your requirement that "**number density as a function of radius should be conserved**", I recommend:

**Go back to Solution 3** (constant cluster):
- It's the only way to get perfect n(r) conservation
- It's physically correct (same cluster, different views)
- It was working perfectly (0.0% error in tests)

The current approach (scaling N but not rs) **cannot** conserve n(r). It's a fundamental mathematical constraint, not a code bug.

## Test Results Summary

| Approach | N galaxies | rs | rmax | n(r) conserved? |
|----------|-----------|-----|------|-----------------|
| Constant cluster | Constant | Constant | Constant | ✅ YES (0.0%) |
| Current (scale N only) | Scales | Constant | Scales | ❌ NO (400-700%) |
| Scale N and rs | Scales | Scales | Scales | ❓ Should work |
| Uniform distribution | Scales | N/A | Scales | ✅ YES |

## The Bottom Line

The current code is working correctly for what it's doing (sampling from NFW). The problem is that **what it's doing is mathematically incompatible with conserving n(r) when fieldsize changes**.

You need to either:
1. Accept that n(r) changes with fieldsize (current behavior)
2. Scale rs with fieldsize (Solution 1)
3. Use uniform distribution instead of NFW (Solution 2)  
4. Keep cluster constant (Solution 3 - previous working version)

