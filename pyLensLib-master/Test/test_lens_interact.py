import pyLensLib.lenstool as lst
from pyLensLib.lens_interact import lens_interact
import argparse

cluster = ['S1063','M0416','M1206pl','PSZ1G311_200',
           'A370','A2744','M0717','M1149',
           'M0329','M1931','M2129','R2129']

# create a dictionary with the cluster names and the corresponding number

cluster_dict = {}
for i in range(len(cluster)):
    cluster_dict[cluster[i]] = i


parser = argparse.ArgumentParser(description='Interactive Lens VISualization Tool')
parser.add_argument("-n","--number", type=int, default=0,help="cluster number. Options are: 0=S1063, 1=M0416, 2=M1206pl, 3=PSZ1G311_200, 4=A370, 5=A2744, 6=M0717, 7=M1149, 8=M0329, 9=M1931, 10=M2129, 11=R2129")
parser.add_argument("-z","--source_redshift", type=float, default=1.0,help="source redshift")
parser.add_argument("-r","--source_re", type=float, default=1.0,help="source effective radius")

args = parser.parse_args()
icl = args.number
zs = args.source_redshift
re = args.source_re



# posizione delle mappe degli angoli di deflessione
path_to_angles = ['/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/stiva/clusters/PSZ1G311/delens/',
                  '/Users/massimo/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/massimo/stiva/pietro_models/',
                  '/Users/massimo/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/massimo/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/',
                  '/Users/massimo/stiva/gabriel_models/best_fit/']


zl_s = lst.getLensRedshift(path_to_angles[icl] + cluster[icl] + '.par')
print (path_to_angles[icl] + cluster[icl] + '.par')
print (('Working with cluster %s at redshift %s') % (cluster[icl], zl_s))
zl = float(zl_s)

print ('creating deflector...')
df = lst.create_deflector(parfile = path_to_angles[icl] + cluster[icl] + '.par',
                          filex = path_to_angles[icl] + cluster[icl] + '_angx.fits',
                          filey = path_to_angles[icl] + cluster[icl] + '_angy.fits',
                          zl= zl, zs = zs, zsnorm = 1.0, resc_fact = 1.0, compute_potential=True)

lims = lst.getFoV(path_to_angles[icl] + cluster[icl] + '.par')

beta1_lim = [lims[0],lims[1]]
beta2_lim = [lims[2],lims[3]]

import matplotlib.pyplot as plt
kwargs_plot = {'beta1_lim': beta1_lim,
               'beta2_lim': beta2_lim,
               'theta1_lim': beta1_lim,
               'theta2_lim': beta2_lim,
               'cluster': cluster,
               'path': path_to_angles}
li = lens_interact(image=None, sersic=True, re=re, with_im_pos=True, **kwargs_plot)
plt.show()