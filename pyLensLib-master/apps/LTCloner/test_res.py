# read outputs of generate_lens_model.py and make some plots

import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from pyLensLib.lenstool import *
from pyLensLib.deflector import deflector
from astropy.cosmology import FlatLambdaCDM
import lenstool
from lenstool.potentials import dpie
from matplotlib.colors import LogNorm

def main():
    parser = argparse.ArgumentParser(description='Test outputs from generate_lens_model.py')
    parser.add_argument('--path', type=str, required=True, help='Path to the directory containing the output files')
    parser.add_argument('--id', type=str, required=True, help='Simulation ID (e.g. 001')
    parser.add_argument('--outfile', type=str, required=True, help='Name of the output image file to be created')
    parser.add_argument("--lens", action="store_true", help="Run lenstool to generate lens model")
    args = parser.parse_args()

    # plot the distribution of the cluster members using circles to represent the galaxies
    # the circle radius is proportional to the galaxy velocity dispersion

    # change directory to the specified path
    os.chdir(args.path)
    # read the data from the output files
    parfile = f'simulated_model_{args.id}.par'

    runmode = readLenstoolBlock(parfile, 'runmode')
    grille = readLenstoolBlock(parfile, 'grille')
    potentiel = readLenstoolBlock(parfile, 'potentiel')
    cline = readLenstoolBlock(parfile, 'cline')
    grande = readLenstoolBlock(parfile, 'grande')
    champ = readLenstoolBlock(parfile, 'champ')
    cosmologie = readLenstoolBlock(parfile, 'cosmologie')

    mainpot = selectPotentielByType(potentiel, ptype='main')
    gals = selectPotentielByType(potentiel, ptype='gal')
    gas = selectPotentielByType(potentiel, ptype='gas')

    fileclmemebers = f'simulated_model_{args.id}_clmemb.csv'
    df_clmemb = pd.read_csv(fileclmemebers, header=0, sep=',')

    fig,ax = plt.subplots(1,2, figsize=(18, 10))
    plotPotentielPositions(mainpot, scale=0.1, ax=ax[0])
    plotPotentielPositions(gals, scale=0.01, ax=ax[1])
    v_disp = np.array([g['v_disp'] for g in gals])
    mag = np.array([g['mag'] for g in gals])
    x_centre = np.array([g['x_centre'] for g in gals])
    y_centre = np.array([g['y_centre'] for g in gals])
    mask = (mag < 24.) & (v_disp > 80.0)
    ax[1].plot(x_centre[mask], y_centre[mask], 'o', color='g', markersize=5, label='Galaxies with mag < 24',alpha=0.5)
    ax[1].plot(df_clmemb['x_centre'], df_clmemb['y_centre'], 'o', color='blue', markersize=5, label='Cluster members', alpha=0.5)
    for i in range(2):
        ax[i].set_xlim(champ['xmin'], champ['xmax'])
        ax[i].set_ylim(champ['ymin'], champ['ymax'])
    ax[0].set_title('Main potentials')
    ax[1].set_title('Galaxies')
    #plt.show()

    # build a deflector object
    co = FlatLambdaCDM(H0=70, Om0=0.3)

    lt = lenstool.Lenstool()

    for g in potentiel:
        lt.add_lens(dpie(g['x_centre'], g['y_centre'], g['ellipticite'], g['angle_pos'],
                         g['z_lens'], g['v_disp'], rc=g['core_radius'], rcut=g['cut_radius']))

    lt.set_cosmology(70, 0.3, 0.7, -1)
    lt.set_field([-100, 100, -100, 100])  # Set the field of view
    lt.set_grid(128, 1)

    conv, wcs = lt.g_mass(1, 2000, gals[0]['z_lens'], 6.0)
    print (conv.min(), conv.max())
    fieldsize = champ['xmax'] - champ['xmin']
    z_lens = gals[0]['z_lens']
    print (f'Field size: {fieldsize}, z_lens: {z_lens}')


    a1, a2, wcs = lt.g_dpl(2000, 6.0)
    kwargs_def = {'zl': z_lens, 'zs': 6.0}
    co = FlatLambdaCDM(H0=cosmologie['H0'], Om0=cosmologie['omegaM'])
    df = deflector(co, angx=a1, angy=a2, **kwargs_def)
    theta = np.linspace(-fieldsize / 2, fieldsize / 2, a1.shape[0])
    df.setGrid(theta=theta)
    tl = df.tancl()
    rl = df.radcl()
    ctl = df.getCaustics(tl)
    crl = df.getCaustics(rl)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(conv, origin='lower', norm=LogNorm(vmin=conv.min(), vmax=conv.max()),
              extent=[-fieldsize / 2, fieldsize / 2, -fieldsize / 2, fieldsize / 2], cmap='viridis')
    ax.contour(conv, levels=np.linspace(0.8, 2.3, 10), colors='white', linewidths=0.5,
               extent=[-fieldsize / 2, fieldsize / 2, -fieldsize / 2, fieldsize / 2])
    for t in tl:
        x, y = df.getCritPoints(t)
        ax.plot(x, y, '-', color='yellow')

    for c in ctl:
        x, y = df.getCausticPoints(c)
        ax.plot(x, y, '-', color='red')

    for t in rl:
        x, y = df.getCritPoints(t)
        ax.plot(x, y, '-', color='yellow')

    for c in crl:
        x, y = df.getCausticPoints(c)
        ax.plot(x, y, '-', color='red')
    #plt.imshow(df.ka, origin='lower', extent=[-champ['xmin'], champ['xmax'], -champ['ymin'], champ['ymax']], cmap='viridis')



    # read source positions from the file
    df_src = pd.read_csv(f'simulated_model_{args.id}_sources.csv', skiprows=1, header=None, sep=' ')
    print (df_src.head())
    df_src.columns = ['id', 'x', 'y', 'a', 'b', 'p', 'z', 'mag']
    # plot the source positions
    ax.plot(df_src['x'], df_src['y'], 'o', color='orange', markersize=5, label='Sources', alpha=0.5)

    print(df.multImaCrossSection(plotUU=True))

    print(df.points_in_UU([903], [906]))
    print(df.points_in_UU([534], [942]))
    print(df.arcsec2pixel([-100.0, 0.0, 100.0], [-100.0, 0.0, 100]))

    plt.show()



    return

if __name__ == '__main__':
    main()

