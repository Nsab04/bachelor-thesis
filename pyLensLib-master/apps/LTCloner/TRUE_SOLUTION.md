# THE TRUE SOLUTION: Constant Cluster, Variable FOV

## The Fundamental Issue

**You CANNOT have all three**:
1. All galaxies within [0, rmax] where rmax scales with fieldsize
2. Number of galaxies N scaling with fieldsize² 
3. Constant number density n(r) at all radii

## Why All Previous Attempts Failed

Any truncation at rmax creates a TRUNCATED NFW distribution, which has different shape from the untruncated NFW. This is mathematical fact, not a code bug.

## The ONLY Working Solution

**Constant Cluster Approach:**
- The cluster has a FIXED size (rmax_cluster) from the input data
- The NFW profile extends to rmax_cluster (fixed)
- fieldsize controls the FOV (what you see), not the cluster size
- More galaxies are generated for clarity, but cluster size is constant

### Implementation:

```python
# Step 1: Determine cluster size from INPUT data
rmax_cluster = np.sqrt(area_input / np.pi)  # From convex hull

# Step 2: Sample galaxies within FIXED cluster size
# Number scales to maintain density
n_galaxies = int(density * π * rmax_cluster² / 3600)

# Step 3: Sample from NFW within rmax_cluster (FIXED!)
r = sample_radius_from_projected_nfw(rs=rs, size=n_galaxies, rmax=rmax_cluster)

# Step 4: fieldsize just controls what you output/visualize
# Galaxies outside fieldsize/2 are still there, just not in FOV
```

### What This Means:

- **fieldsize=100**: You see the inner 50" of a cluster that extends to (say) 150"
- **fieldsize=400**: You see 200" of the same cluster
- **The cluster itself doesn't change**
- **Density n(r) is perfectly preserved**

## Alternative: Accept Non-Conservation

If you MUST have:
- Galaxies only within fieldsize/2×√2
- More galaxies for larger fieldsize

Then you MUST accept:
- **Density n(r) WILL change** (this is unavoidable mathematics)
- Inner regions will have higher density for larger fieldsize
- This is the inherent property of truncated NFW

## Recommendation

Use the **Constant Cluster** approach. It's:
- ✅ Physically correct (cluster has a size)
- ✅ Mathematically sound (density conserved)
- ✅ Computationally efficient (no rejection needed)
- ✅ Interpretable (fieldsize = FOV, not cluster size)

The alternative (scaling rmax with fieldsize) cannot maintain constant density - this is proven mathematics, not a code issue.

