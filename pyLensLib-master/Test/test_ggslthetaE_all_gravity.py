import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt

import pyLensLib.lenstool as lst
import tqdm

"""
File location on gravity:
/home/meneghetti/stiva/lensmodels/

Subfolders (cluster models:
- A2744
- AS1063
- elgordo
- M0416
- M0416_2022
- M1206
- PLCK-G287
- PSZ1G311  

In each subfolder, a best.par file can be read to instantiate a lenstool object.

Each subfolder, contains a tmp/ subfolder with multiple deflection angle maps in fits format:
E.g.,  defX_z*.fits and defY_z*.fits

This script processes all clusters and compute the GGSL cross section and Einstein radius for each cluster
"""

import glob
import numpy as np

def processCluster(base_path, cluster_name, output_file):

    parfile = f'{base_path}{cluster_name}/best.par'

    defX_files = sorted(glob.glob(f'{base_path}{cluster_name}/tmp/defX_z*.fits'))
    defY_files = sorted(glob.glob(f'{base_path}{cluster_name}/tmp/defY_z*.fits'))

    # Validate file lists
    if not defX_files or not defY_files:
        print(f"Warning: No deflection files found for cluster {cluster_name}")
        return

    if len(defX_files) != len(defY_files):
        print(
            f"Error: Mismatched file counts for {cluster_name}: {len(defX_files)} defX vs {len(defY_files)} defY files")
        return

    # Validate matching z*.fits suffixes
    for defX_file, defY_file in zip(defX_files, defY_files):
        defX_suffix = defX_file.split('defX_')[-1]
        defY_suffix = defY_file.split('defY_')[-1]
        if defX_suffix != defY_suffix:
            print(f"Error: Mismatched suffixes for {cluster_name}: {defX_suffix} vs {defY_suffix}")
            return


    x1, x2 = lst.getClMembDelimitingPoints(parfile)
    zl = lst.getLensRedshift(parfile)

    zs_arr = [1.0, 3.0, 6.0]
    for zs in zs_arr:
        thetaE_list = []
        gssl_list = []
        compute_fovsp = True
        for defX_file, defY_file in zip(defX_files, defY_files):


            lst_deflector = lst.create_deflector(parfile, defX_file, defY_file, zl=zl, zs=zs,
                                                usePotential=False, zsnorm=1.0, resc_fact=1.0,
                                                compute_potential=False)

            if compute_fovsp:
                _, _, fov_sp = lst_deflector.fovSP_from_x1x2(x1, x2)
                compute_fovsp = False
            gssl = lst_deflector.ggslCrossSection()
            thetaE = lst_deflector.thetaE()
            thetaE_list.append(thetaE)
            gssl_list.append(gssl)

        thetaE_median = float(np.median(thetaE_list))
        thetaE_1st_quartile = float(np.percentile(thetaE_list, 25))
        thetaE_3rd_quartile = float(np.percentile(thetaE_list, 75))
        ggsl_median = float(np.median(gssl_list))
        ggsl_1st_quartile = float(np.percentile(gssl_list, 25))
        ggsl_3rd_quartile = float(np.percentile(gssl_list, 75))
        pggsl_median = float(ggsl_median)/fov_sp
        pggsl_1st_quartile = float(ggsl_1st_quartile)/fov_sp
        pggsl_3rd_quartile = float(ggsl_3rd_quartile)/fov_sp

        with open(output_file, 'a') as f:
            values = [cluster_name, zl, zs, thetaE_median, thetaE_1st_quartile, thetaE_3rd_quartile,
                      ggsl_median, ggsl_1st_quartile, ggsl_3rd_quartile,
                      pggsl_median*1e6, pggsl_1st_quartile*1e6, pggsl_3rd_quartile*1e6]
            f.write(','.join(map(str, values)) + '\n')

        ax.errorbar(pggsl_median * 1e6, thetaE_median,
                    xerr=np.array(
                        [[(pggsl_median - pggsl_1st_quartile) * 1e6], [(pggsl_3rd_quartile - pggsl_median) * 1e6]]),
                    yerr=np.array(
                        [[(thetaE_median - thetaE_1st_quartile)], [(thetaE_3rd_quartile - thetaE_median)]]),
                    fmt='o', label=f'{cluster_name} (z_s={zs})')


clusters = ['A2744', 'AS1063', 'elgordo', 'M0416', 'M0416_2022', 'M1206', 'PLCK-G287', 'PSZ1G311']
base_path = '/home/meneghetti/stiva/lensmodels/'
output_file = 'ggsl_thetaE_results.txt'

fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.set_xlabel(r'P_{GGSL}')
ax.set_ylabel(r'$\theta_E$ [arcsec]')

# open file to write results
with open(output_file, 'w') as f:
    f.write('Cluster,z_l,z_s,thetaE_median,thetaE_1st_quartile,thetaE_3rd_quartile,'
            'ggsl_median,ggsl_1st_quartile,ggsl_3rd_quartile,'
            'pggsl_median,pggsl_1st_quartile,pggsl_3rd_quartile\n')
for cluster in clusters:
    print(f'Processing cluster: {cluster}')
    # create a lenstool object
    parfile = f'{base_path}{cluster}/best.par'
    processCluster(base_path, cluster, output_file)

# close file


ax.legend()
fig.savefig('ggsl_thetaE_all_gravity.png')