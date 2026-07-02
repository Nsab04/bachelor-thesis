"""
This code reads an input lenstool file and generates a deflector object. For a set of
source redshift, it computes the cross sections for ggsl and multiple images, the FOV on the source plane,
and the Einstein radius.
Then it reads all the realizations of the same lens in another directory and produces plots comparing
the different lensing properties.

Inputs:
    - lenstool_input: lenstool input file
    - lenstool_input_ax: deflection angle map input file (ax)
    - lenstool_input_ay: deflection angle map input file (ay)
    - lens_realizations_dir: directory containing the different realizations of the same lens
"""


import numpy as np
import matplotlib.pyplot as plt
import pyLensLib.lenstool as lst
from pyLensLib.deflector import deflector

import argparse
import astropy.io.fits as fits
from astropy.cosmology import FlatLambdaCDM
import lenstool
from lenstool.potentials import dpie

# parse arguments
parser = argparse.ArgumentParser(description='Compare lensing properties of different lens realizations.')
parser.add_argument('lenstool_input', type=str, help='Path to the lenstool input file.')
parser.add_argument('lenstool_input_ax', type=str, help='Path to the lenstool ax file.')
parser.add_argument('lenstool_input_ay', type=str, help='Path to the lenstool ay file.')
parser.add_argument('lens_realizations_dir', type=str, help='Directory containing lens realizations.')
args = parser.parse_args()

zcl = lst.getLensRedshift(args.lenstool_input)
x1, x2 = lst.getClMembDelimitingPoints(args.lenstool_input)
angx = fits.open(args.lenstool_input_ax)[0].data
angy = fits.open(args.lenstool_input_ay)[0].data

df = lst.create_deflector(args.lenstool_input,
                        filex=args.lenstool_input_ax,
                        filey=args.lenstool_input_ay,
                        zl=zcl,zsnorm=1.0,zs=1.0)

zs_ggsl = np.linspace(1.0, 6.0, 10)
_cs = []
for z in zs_ggsl:
    df.change_redshift(z)
    _,_,fov_sp = df.fovSP_from_x1x2(x1, x2)
    _cs.append({'z': z,
                'ggsl_cs': df.ggslCrossSection(minsize=0.5),
                'multima_cs': df.multImaCrossSection(),
                'fov_sp': fov_sp,
                'thetaE': df.thetaE()})

# check the _cross_sections.csv  files in the lens_realizations_dir
import os
resu_files = [f for f in os.listdir(args.lens_realizations_dir) if f.endswith('_cross_sections.csv')]

fig_cs, ax_cs = plt.subplots(1,3, figsize=(20,6))

for rf in resu_files:
    # extract the model name from the file name
    # the file name is simulated_model_<model name>_cross_sections.csv
    model_name = rf.split('_cross_sections.csv')[0].replace('simulated_model_', '')
    filepath = os.path.join(args.lens_realizations_dir, rf)
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    #ax_cs[0].plot(data['z'], data['ggsl_cs']/data['fov_sp']*1e6, marker='o', label=model_name, alpha=0.5)
    ax_cs[0].plot(data['z'], data['ggsl_cs'], marker='o', label=model_name, alpha=0.5)
    ax_cs[1].plot(data['z'], data['multima_cs'], marker='o', alpha=0.5)
    ax_cs[2].plot(data['z'], data['thetaE'], marker='o', label=model_name, alpha=0.5)

z_vals = [item['z'] for item in _cs]
cs_vals = [item['ggsl_cs'] for item in _cs]
mi_vals = [item['multima_cs'] for item in _cs]
fov_sp = [item['fov_sp'] for item in _cs]
#ax_cs[0].plot(z_vals, np.array(cs_vals)/np.array(fov_sp)*1e6, marker='o',label="Input", color='black', linewidth=2)
ax_cs[0].plot(z_vals, np.array(cs_vals), marker='o',label="Input", color='black', linewidth=2)
ax_cs[0].legend()
ax_cs[0].set_xlabel("Source Redshift")
ax_cs[0].set_ylabel("GGSL Cross-Section (arcsec²)")
ax_cs[0].set_title(f"GGSL Cross-Section vs Redshift for input Model")

ax_cs[1].plot(z_vals, np.array(mi_vals), marker='o',label="Input Model", color='black', linewidth=2)
ax_cs[1].set_xlabel("Source Redshift")
ax_cs[1].set_ylabel("MI Cross-Section (arcsec²)")
ax_cs[1].set_title(f"MI Cross-Section vs Redshift for input Model")

thetaE_vals = [item['thetaE'] for item in _cs]
ax_cs[2].plot(z_vals, np.array(thetaE_vals), marker='o',label="Input Model", color='black', linewidth=2)
ax_cs[2].set_xlabel("Source Redshift")
ax_cs[2].set_ylabel("Einstein Radius (arcsec)")
ax_cs[2].set_title(f"Einstein Radius vs Redshift for input Model")
for i in range(3):
    #ax_cs[i].set_yscale('log')
    ax_cs[i].grid()
fig_cs.tight_layout()
fig_cs.savefig('cross_sections_comparison.png', dpi=150)
plt.tight_layout()
plt.show()

# now compare the convergence maps: how much they differ from the input model? Assume zs=6.0

df.change_redshift(6.0)
kappa_input = df.ka
int_ka = np.nansum(kappa_input)
print(df.ka.shape)

test_kappa = False

if test_kappa:
    resu_files = [f for f in os.listdir(args.lens_realizations_dir) if f.endswith('.par')]
    ka_rat = []
    for rf in resu_files:
        filepath = os.path.join(args.lens_realizations_dir, rf)
        runmode = lst.readLenstoolBlock(filepath, 'runmode')
        grille = lst.readLenstoolBlock(filepath, 'grille')
        potentiel = lst.readLenstoolBlock(filepath, 'potentiel')
        cline = lst.readLenstoolBlock(filepath, 'cline')
        grande = lst.readLenstoolBlock(filepath, 'grande')
        champ = lst.readLenstoolBlock(filepath, 'champ')
        cosmologie = lst.readLenstoolBlock(filepath, 'cosmologie')
        lt = lenstool.Lenstool()
        lt.set_cosmology(cosmologie['H0'], cosmologie['omegaM'], cosmologie['omegaX'], cosmologie['wX'])
        lt.set_field([champ['xmin'], champ['xmax'], champ['ymin'], champ['ymax']])
        for g in potentiel:
            lt.add_lens(dpie(g['x_centre'], g['y_centre'], g['ellipticite'], g['angle_pos'],
                             g['z_lens'], g['v_disp'], rc=g['core_radius'], rcut=g['cut_radius']))
        lt.set_grid(128, 1)
        z_lens = potentiel[0]['z_lens']
        a1, a2, wcs = lt.g_dpl(2000, 1.0)
        kwargs_def = {'zl': z_lens, 'zs': 1.0}
        co = FlatLambdaCDM(H0=cosmologie['H0'], Om0=cosmologie['omegaM'])
        df_ = deflector(co, angx=a1, angy=a2, **kwargs_def)
        theta = np.linspace(champ['xmin'], champ['xmax'], a1.shape[0])
        df_.setGrid(theta=theta)
        df_.change_redshift(6.0)
        ka_rat.append(np.nansum(df_.ka)/int_ka)

    fig_ka, ax_ka = plt.subplots(1,1, figsize=(8,6))
    ax_ka.hist(ka_rat, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax_ka.axvline(1.0, color='red', linestyle='dashed', linewidth=2, label='Input Model')
    ax_ka.set_xlabel('Total Convergence Ratio (Realization / Input Model)')
    ax_ka.set_ylabel('Number of Realizations')
    ax_ka.set_title('Distribution of Total Convergence Ratios at zs=6.0')
    ax_ka.legend()
    fig_ka.tight_layout()
    fig_ka.savefig('kappa_ratio_histogram.png', dpi=150)
    plt.tight_layout()
    plt.show()





