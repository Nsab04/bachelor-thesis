# NFW Profile Terminology Clarification

## Question
What does "density" mean in the plot_projected_nfw.py script? Is it mass inside a radius or in an annulus?

## Answer

The terminology refers to **different but related quantities**:

### 1. **f(x) = _f_proj_nfw(x)** 
**CUMULATIVE MASS** within radius R
- Definition: f(R) = ∫₀ᴿ Σ(R') × 2πR' dR'
- Units: Total mass
- Meaning: **Mass INSIDE a circle** of radius R

### 2. **Σ(R) = df/dR**
**SURFACE DENSITY** at radius R
- Definition: df/dR (derivative of cumulative mass)
- Units: Mass per unit area
- Meaning: **Mass per unit area AT radius R** (in an infinitesimally thin annulus)

### 3. **dM = Σ(R) × 2πR dR**
**MASS IN AN ANNULUS** between R and R+dR
- Definition: Surface density × circumference × width
- Units: Mass
- Meaning: **Mass IN an annulus** at radius R

### 4. **df/dx × x** (dimensionless)
**MASS PER LOGARITHMIC BIN** = dM/d(ln R)
- Definition: In dimensionless units (x = R/rs), this is df/dx × x
- Proportional to: Σ(R) × 2πR
- Meaning: Mass in annulus at R (for logarithmic binning)

## For Galaxy Sampling

When we sample galaxy positions from the NFW profile:

```python
# Sample from CDF
u = random.uniform(0, 1, N)
r = interpolate(u, cdf_values, r_values)
```

This is **equivalent** to sampling from a PDF:
```
PDF(R) ∝ Σ(R) × 2πR ∝ df/dR × R
```

**What this means:**
- Probability of a galaxy landing in an annulus at radius R is proportional to **mass in that annulus**
- NOT proportional to surface density alone
- NOT the total mass inside R

## Visualization in the Script

### Plot 2, Left Panel: "Surface Density Σ(R)"
- Shows: **df/dx** (mass per unit area)
- This is the density **AT** radius R
- **NOT** the mass inside R

### Plot 2, Right Panel: "Mass in Annulus"
- Shows: **df/dx × x** (mass per logarithmic bin)
- This is proportional to mass **IN** an annulus at R
- This is what determines the probability for galaxy sampling

## Summary Table

| Quantity | Symbol | Meaning | Radius vs Annulus |
|----------|--------|---------|------------------|
| Cumulative Mass | f(R) | Total mass within R | **Inside radius R** |
| Surface Density | Σ(R) = df/dR | Mass per unit area | **At radius R** (infinitesimal annulus) |
| Mass in Annulus | dM = Σ×2πR×dR | Mass in finite annulus | **In annulus** at R |
| Sampling PDF | ∝ Σ×2πR | Probability distribution | Proportional to **mass in annulus** |

## Key Insight

When we say "number density n(r)" should be constant, we mean:
- **Number of galaxies per unit area AT radius r** should be constant
- This corresponds to **surface density** Σ(r)
- NOT the total number inside r (which obviously increases with r)

The hybrid approach maintains constant **surface density** Σ(r) by keeping the cluster size fixed.

---
Updated: January 23, 2026

