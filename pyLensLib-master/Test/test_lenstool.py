from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import pyLensLib.lenstool as lst
from pyLensLib.lenstool import *

df = lst.getClMembers('/Users/maxmen3/stiva/clusters/PSZ1G311/delens/best.par')
df.to_csv('PSZ1G311_cat_pars.csv',sep=' ')

df = lst.getClMembers('/Users/maxmen3/stiva/pietro_models/M1206pl.par')
df.to_csv('M1206pl.csv',sep=' ')

print (lst.getLensRedshift('/Users/maxmen3/stiva/clusters/PSZ1G311/delens/best.par'))

runmode = readLenstoolBlock(best_par='/Users/maxmen3/stiva/pietro_models/M1206pl.par',block_name='runmode')
grille = readLenstoolBlock(best_par='/Users/maxmen3/stiva/pietro_models/M1206pl.par',block_name='grille')
potentiel = readLenstoolBlock(best_par='/Users/maxmen3/stiva/pietro_models/M1206pl.par',block_name='potentiel')
cline = readLenstoolBlock(best_par='/Users/maxmen3/stiva/pietro_models/M1206pl.par',block_name='cline')
grande = readLenstoolBlock(best_par='/Users/maxmen3/stiva/pietro_models/M1206pl.par',block_name='grande')
cosmologie = readLenstoolBlock(best_par='/Users/maxmen3/stiva/pietro_models/M1206pl.par',block_name='cosmologie')
champ = readLenstoolBlock(best_par='/Users/maxmen3/stiva/pietro_models/M1206pl.par',block_name='champ')

#for pot in potentiel:
#    print (pot)

print (runmode)

writeLenstoolBlock("output.par", "runmode", runmode)
writeLenstoolBlock("output.par", "grille", grille, append=True)
writeLenstoolBlock("output.par", "potentiel", potentiel, append=True)
writeLenstoolBlock("output.par", "cline", cline, append=True)
writeLenstoolBlock("output.par", "grande", grande, append=True)
writeLenstoolBlock("output.par", "cosmologie", cosmologie, append=True)
writeLenstoolBlock("output.par", "champ", champ, finalize=True, append=True)


# loop over the "potentiel" blocks. Read the values of keywords mag, v_disp, cut_radius, and core radius.

mag = []
v_disp = []
cut_radius = []
core_radius = []
for pot in potentiel:
    x = findInBlock(pot, 'x_centre')
    y = findInBlock(pot, 'y_centre')
    mag_ = findInBlock(pot, 'mag')
    v_disp_ = findInBlock(pot, 'v_disp')
    cut_radius_ = findInBlock(pot, 'cut_radius')
    core_radius_ = findInBlock(pot, 'core_radius')
    if mag_ is not None and v_disp_ is not None and cut_radius_ is not None and core_radius_ is not None:
        print (f"mag: {mag_}, v_disp: {v_disp_}, cut_radius: {cut_radius_}, core_radius: {core_radius_}")
        mag.append(mag_)
        v_disp.append(v_disp_)
        cut_radius.append(cut_radius_)
        core_radius.append(core_radius_)

mag = np.array(mag)
v_disp = np.array(v_disp)
cut_radius = np.array(cut_radius)
core_radius = np.array(core_radius)

mag0 = mag[0]
v_disp_0 = v_disp[0]
core_radius_0 = core_radius[0]
cut_radius_0 = cut_radius[0]

# plot the sigma-luminosity relation

import numpy as np
def plot_fit(mag, v_disp, v_disp_noisy, alpha, ax=None, xlabel="Magnitude", ylabel="Velocity Dispersion"):
    """
    Plot data and fitted v_disp-mag relation, skipping NaNs and None.
    """
    mag = np.array(mag, dtype=float)
    v_disp = np.array(v_disp, dtype=float)

    # Filter valid entries
    valid = ~np.isnan(mag) & ~np.isnan(v_disp)
    mag = mag[valid]
    v_disp = v_disp[valid]

    if len(mag) < 2:
        raise ValueError("Not enough valid data to plot.")

    mag0 = mag[0]
    v0 = v_disp[0]

    model = lambda m: v0 * 10 ** (-0.4 * alpha * (m - mag0))
    m_vals = np.linspace(min(mag), max(mag), 100)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(mag, v_disp_noisy, 'o', label='Data')
    ax.plot(m_vals, model(m_vals), '-', label=f'Fit (α = {alpha:.2f})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

########## SCALING RELATIONS ##########

mag, v_disp, cut_radius, core_radius, alpha, beta = measureScalingRelations(potentiel)
print (f"Fitted alpha: {alpha}, beta: {beta}")
#plot_fit(mag, v_disp, alpha)
fig, ax = plt.subplots(1,2, figsize=(10, 5))
v_disp_noisy = addScatterToScalingRelation(mag, alpha, v_disp[0], x0=mag[0], scatter=0.05, seed=None, useLogNormal=True)
plot_fit(mag, v_disp, v_disp_noisy, alpha, ax=ax[0], xlabel="Magnitude", ylabel="Velocity Dispersion [km/s]")
cut_radius_noisy = addScatterToScalingRelation(mag, beta, cut_radius[0], x0=mag[0], scatter=0.05, seed=None, useLogNormal=True)
plot_fit(mag, cut_radius, cut_radius_noisy, beta, ax=ax[1], xlabel="Magnitude", ylabel="Cut Radius [kpc]")

plt.show()


###### radial distribution of cluster members #######
r, density, counts, err_density, err_counts = measureRadialDistribution(potentiel, rmin=0, rmax=75)

maxdist=0.0
for pot in potentiel:
    x = findInBlock(pot, 'x_centre')
    y = findInBlock(pot, 'y_centre')
    v = findInBlock(pot, 'v_disp')
    mag = findInBlock(pot, 'mag')
    if x is not None and y is not None and v is not None and mag is not None:
        dist = np.sqrt(x**2 + y**2)
        if dist > maxdist:
            maxdist = dist

print(f"Maximum distance of cluster members: {maxdist:.2f} arcsec")

# Fit
popt, pcov = fitNFWtoRadialDistribution(r, density, err_density)
n0, rs = popt
print(f"Best-fit NFW parameters: n0 = {n0:.3f}, rs = {rs:.3f}")

# Plot
plt.errorbar(r, density, yerr=err_density, fmt='o', label='Measured')
r_fit = np.linspace(r.min(), r.max(), 300)
plt.plot(r_fit, projected_NFW(r_fit, *popt), '-', label='NFW Fit')
plt.xlabel("Radius")
plt.ylabel("Surface number density")
plt.legend()
plt.grid(True)
plt.show()


#### Luminosity function of cluster members ####
area_m1206 = (maxdist*2.0/np.sqrt(2.0))**2/3600.0  # area in arcsec^2, assuming a square field of view
mag_centers_lf, counts_lf, err_counts_lf = measureLuminosityFunction(potentiel, mag_min=17.0, mag_max=24.0, nbins=10, area=area_m1206)
deltamag = mag_centers_lf[1] - mag_centers_lf[0]

mainpot=selectPotentielByType(potentiel, ptype='main')
gals=selectPotentielByType(potentiel, ptype='gal')
gas=selectPotentielByType(potentiel, ptype='gas')

print ((counts*deltamag*area_m1206).sum(),len(gals))
popt_s, pcov_s = fitSchechterFunction(mag_centers_lf, counts_lf, err_counts_lf)
phistar, Mstar, alpha_lf = popt_s

magsampled = sampleMagnitudesFromFittedLF(phistar, Mstar, alpha_lf,
                                 24, 26,
                                 area=area_m1206,
                                 scatter=0.0,
                                 seed=None,
                                 bins=5000)

bins = np.linspace(24.0, 26.0, 4)
counts2_lf, edges2_lf = np.histogram(magsampled, bins=bins)
err_counts2_lf = np.sqrt(counts2_lf)

mag_centers2_lf = 0.5 * (edges2_lf[1:] + edges2_lf[:-1])
binsize = edges2_lf[2] - edges2_lf[1]
print (mag_centers2_lf, counts2_lf)

id_last = gals[-1]['id']
zlens = gals[-1]['z_lens']
print(f"Last galaxy ID in the list: {id_last}")
print (mag0, v_disp_0, cut_radius_0, core_radius_0)
for m in magsampled:
    newgalid = str(int(id_last) + 1)
    newpot = createPotentielBlock(newgalid,m,rs=rs,zlens=zlens, alpha=alpha, beta=beta, mag_0=mag0, v_disp_0=v_disp_0, cut_radius_0=cut_radius_0, core_radius_0=core_radius_0, rmax=100.0*np.sqrt(2.0), seed=42)
    #print(f"Generated galaxy {newgalid}: {m:.2f}, v_disp={newpot['v_disp']:.2f}, cut_radius={newpot['cut_radius']:.2f}, core_radius={newpot['core_radius']:.2f}")
    gals.append(newpot)

# Plotting
M_fit = np.linspace(17, 30, 300)
phi_fit = schechter_mag(M_fit, *popt_s)

plt.errorbar(mag_centers_lf, counts_lf, yerr=err_counts_lf, fmt='o', label='Data')
plt.errorbar(mag_centers2_lf, counts2_lf/binsize/area_m1206, yerr=err_counts2_lf/binsize/area_m1206, fmt='o', label='Data added')
plt.plot(M_fit, phi_fit, 'r-', label='Schechter fit')
plt.gca().invert_xaxis()
plt.yscale('log')
plt.xlabel("Magnitude")
plt.ylabel("Galaxy count per unit mag and area")
plt.title("Luminosity Function with Schechter Fit")
plt.legend()
plt.show()

print(f"Best-fit parameters:\n  phi* = {popt_s[0]:.3e}, M* = {popt_s[1]:.2f}, alpha_lf = {popt_s[2]:.2f}")

#exit()

#### visualization of the cluster and its members ####

from astropy.io import fits
import os
fig, ax = plt.subplots(1,1, figsize=(10, 10))
if os.path.exists('/Users/maxmen3/stiva/pietro_models/M1206pl_rgb.fits'):
    with fits.open('/Users/maxmen3/stiva/pietro_models/M1206pl_rgb.fits') as hudl:
        rgb_fits_data = hudl[0].data
    rgb_image = np.transpose(rgb_fits_data, (1, 2, 0))
    rgb_image = rgb_image / np.max(rgb_image)

    ax.imshow(rgb_image, origin='lower', aspect='auto',extent=[-100, 100, -100, 100])



#

mainpot_ordered, gals_ordered =  matchPotentielListsByDistance(mainpot, gals)
#i = 0
#for p in mainpot_ordered:
#    g = gals_ordered[i]
#    print(f"Main potential: {p['id']}, v_disp={p['v_disp']} Associated galaxy: {g['id']}, v_disp={g['v_disp']}, x_centre={g['x_centre']}, y_centre={g['y_centre']}, ellipticite={g['ellipticite']}, angle_pos={g['angle_pos']}, z_lens={g['z_lens']}, core_radius={g['core_radius']}, cut_radius={g['cut_radius']}")
#    i+= 1

# concatenate the mainpot_ordered and gals_ordered[0:len(mainpot_ordered)] lists
gals_ = mainpot_ordered + gals_ordered[0:len(mainpot_ordered)] + gas
gals_all = mainpot_ordered + gals_ordered + gas

plotPotentielPositions(gals_,scale=0.01,ax=ax)
plt.show()


# generate a lenstool model with the mainpot_ordered and gals_ordered lists

import lenstool
from lenstool.potentials import dpie
lt = lenstool.Lenstool()

for g in gals_all:
    lt.add_lens(dpie(g['x_centre'],g['y_centre'],g['ellipticite'],g['angle_pos'],
                     g['z_lens'],g['v_disp'],rc=g['core_radius'], rcut=g['cut_radius']))

lt.set_cosmology(70, 0.3, 0.7, -1)
lt.set_field([-100, 100, -100, 100])  # Set the field of view
lt.set_grid(1000, 1)

conv, wcs = lt.g_mass(1, 1000, g['z_lens'], 6.0)
a1, a2, wcs = lt.g_dpl(2000,6.0)

fig,ax = plt.subplots(3,3, figsize=(12, 12))
ax[0,0].imshow(conv, origin='lower', cmap='viridis',norm=LogNorm(vmin=conv.min(), vmax=conv.max()),extent=[-100, 100, -100, 100])
ax[0,0].contour(conv, levels=np.linspace(0.8, 2.3, 10), colors='white', linewidths=0.5,extent=[-100, 100, -100, 100])
ax[0,0].set_aspect('equal')
ax[0,0].set_xlim(-100, 100)
ax[0,0].set_ylim(-100, 100)

#### generate multiple models inspired to the original one with different seeds and plot them


for i in range(3):
    for j in range(3):
        if i == 0 and j == 0:
            continue
        else:
            seed = 12 + i * 3 + j
            model_smooth, model_gal, model_gas = generateLenstoolModel(mainpot_ordered, gals_ordered, gas, rs, tolerance=0.8, offset=1.0, scatter=0.05, useLogNormal=True, opening_angle=30.0, seed=seed)
            lt2 = lenstool.Lenstool()
            model_total = model_smooth + model_gal + model_gas
            for g in model_total:
                lt2.add_lens(dpie(g['x_centre'], g['y_centre'], g['ellipticite'], g['angle_pos'],
                                  g['z_lens'], g['v_disp'], rc=g['core_radius'], rcut=g['cut_radius']))
            lt2.set_cosmology(70, 0.3, 0.7, -1)
            lt2.set_field([-100, 100, -100, 100])  # Set the field of view
            lt2.set_grid(1000, 1)
            conv2, wcs2 = lt2.g_mass(1, 1000, g['z_lens'], 6.0)
            ax[i,j].imshow(conv2, origin='lower', cmap='viridis',norm=LogNorm(vmin=conv2.min(), vmax=conv2.max()),extent=[-100, 100, -100, 100])
            ax[i,j].contour(conv2, levels=np.linspace(0.8, 2.3, 10), colors='white', linewidths=0.5,extent=[-100, 100, -100, 100])
            plotPotentielPositions(model_smooth+model_gas, scale=0.01, ax=ax[i,j])
            ax[i,j].set_aspect('equal')
            ax[i,j].set_xlim(-100, 100)
            ax[i,j].set_ylim(-100, 100)

fig.tight_layout()
plt.show()

# radial distribution of cluster members (from the last generated model)
r2, density2, counts2, err_density2, err_counts2 = measureRadialDistribution(model_total, rmin=0, rmax=75)

# Fit
popt2, pcov2 = fitNFWtoRadialDistribution(r2, density2, err_density2)
n02, rs2 = popt2

print(f"Best-fit NFW parameters: n0 = {n02:.3f}, rs = {rs2:.3f}")

# Plot
plt.errorbar(r, density, yerr=err_density, fmt='o', label='Measured')
r_fit = np.linspace(r.min(), r.max(), 300)
plt.plot(r_fit, projected_NFW(r_fit, *popt), '-', label='NFW Fit')
plt.errorbar(r2, density2, yerr=err_density2, fmt='o', label='Measured (new model)')
plt.plot(r_fit, projected_NFW(r_fit, *popt2), '--', label='NFW Fit (new model)')
plt.xlabel("Radius")
plt.ylabel("Surface number density")
plt.legend()
plt.grid(True)
plt.show()


# models from scratch

# compute the number of galaxies to be genereated in a field of view of 100x100 arcsec
area = 200 * 200 / 3600.0  # in square arcmin

print (f"Area of the field of view: {area:.2f} square arcmin")

magsampled = sampleMagnitudesFromFittedLF(phistar, Mstar, alpha_lf,
                                 17, 26,
                                 area=area,
                                 scatter=0.0,
                                 seed=None,
                                 bins=5000)

bins = np.linspace(17.0, 26.0, 10)
counts3_lf, edges3_lf = np.histogram(magsampled, bins=bins)
err_counts3_lf = np.sqrt(counts3_lf)

mag_centers3_lf = 0.5 * (edges3_lf[1:] + edges3_lf[:-1])
binsize3 = edges3_lf[2] - edges3_lf[1]
print (len(mag_centers3_lf), len(counts3_lf), len(err_counts3_lf), binsize3)

plt.errorbar(mag_centers_lf, counts_lf, yerr=err_counts_lf, fmt='o', label='Data')
plt.errorbar(mag_centers2_lf, counts2_lf/binsize/area_m1206, yerr=err_counts2_lf/binsize/area_m1206, fmt='o', label='Data added')
plt.errorbar(mag_centers3_lf, counts3_lf/binsize3/area, yerr=err_counts3_lf/binsize3/area, fmt='o', label='Data generated')
plt.plot(M_fit, phi_fit, 'r-', label='Schechter fit')
plt.gca().invert_xaxis()
plt.yscale('log')
plt.xlabel("Magnitude")
plt.ylabel("Galaxy count per unit mag and area")
plt.title("Luminosity Function with Schechter Fit")
plt.legend()
plt.show()

#exit()

print (f"Number of galaxies to be generated: {len(magsampled)}")
gals_new = []
for i, m in enumerate(magsampled):
    newgalid = str(i + 10000)
    newpot = createPotentielBlock(newgalid, m, rs=rs, zlens=zlens, alpha=alpha, beta=beta, mag_0=mag0, v_disp_0=v_disp_0,
                         cut_radius_0=cut_radius_0, core_radius_0=core_radius_0, rmax=100.0*np.sqrt(2.0), seed=42)
    print (f"Generated galaxy {newgalid}: {m:.2f}, v_disp={newpot['v_disp']:.2f}, cut_radius={newpot['cut_radius']:.2f}, core_radius={newpot['core_radius']:.2f}")
    gals_new.append(newpot)

gals_new[0] = gals[0]
print ('Number of galaxies generated: ', len(gals_new))
#exit()

# generate a lenstool model with the gals_new list
fig,ax = plt.subplots(3,3, figsize=(12, 12))
ax[0,0].imshow(conv, origin='lower', cmap='viridis',norm=LogNorm(vmin=conv.min(), vmax=conv.max()),extent=[-100, 100, -100, 100])
ax[0,0].contour(conv, levels=np.linspace(0.8, 2.3, 10), colors='white', linewidths=0.5,extent=[-100, 100, -100, 100])
ax[0,0].set_aspect('equal')
ax[0,0].set_xlim(-100, 100)
ax[0,0].set_ylim(-100, 100)

#### generate multiple models inspired to the original one with different seeds and plot them


for i in range(3):
    for j in range(3):
        if i == 0 and j == 0:
            continue
        else:
            seed = 12 + i * 3 + j
            model_smooth, model_gal, model_gas = generateLenstoolModel(mainpot_ordered, gals_new, gas, rs, rmax=100.0*np.sqrt(2.0), tolerance=0.8, offset=1.0, scatter=0.05, useLogNormal=True, opening_angle=30.0, seed=seed)
            lt2 = lenstool.Lenstool()
            model_total = model_smooth + model_gal + model_gas
            for g in model_total:
                lt2.add_lens(dpie(g['x_centre'], g['y_centre'], g['ellipticite'], g['angle_pos'],
                                  g['z_lens'], g['v_disp'], rc=g['core_radius'], rcut=g['cut_radius']))
            lt2.set_cosmology(70, 0.3, 0.7, -1)
            lt2.set_field([-100, 100, -100, 100])  # Set the field of view
            lt2.set_grid(1000, 1)
            conv2, wcs2 = lt2.g_mass(1, 1000, g['z_lens'], 6.0)
            ax[i,j].imshow(conv2, origin='lower', cmap='viridis',norm=LogNorm(vmin=conv2.min(), vmax=conv2.max()),extent=[-100, 100, -100, 100])
            ax[i,j].contour(conv2, levels=np.linspace(0.8, 2.3, 10), colors='white', linewidths=0.5,extent=[-100, 100, -100, 100])
            plotPotentielPositions(model_smooth+model_gas, scale=0.01, ax=ax[i,j])
            ax[i,j].set_aspect('equal')
            ax[i,j].set_xlim(-100, 100)
            ax[i,j].set_ylim(-100, 100)

fig.tight_layout()
plt.show()

# radial distribution of cluster members (from the last generated model)
r2, density2, counts2, err_density2, err_counts2 = measureRadialDistribution(model_total, rmin=0, rmax=75)

# Fit
popt2, pcov2 = fitNFWtoRadialDistribution(r2, density2, err_density2)
n02, rs2 = popt2

print(f"Best-fit NFW parameters: n0 = {n02:.3f}, rs = {rs2:.3f}")

# Plot
plt.errorbar(r, density, yerr=err_density, fmt='o', label='Measured')
r_fit = np.linspace(r.min(), r.max(), 300)
plt.plot(r_fit, projected_NFW(r_fit, *popt), '-', label='NFW Fit')
plt.errorbar(r2, density2, yerr=err_density2, fmt='o', label='Measured (new model)')
plt.plot(r_fit, projected_NFW(r_fit, *popt2), '--', label='NFW Fit (new model)')
plt.xlabel("Radius")
plt.ylabel("Surface number density")
plt.legend()
plt.grid(True)
plt.show()