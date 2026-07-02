"""
This code generates nlens lenses and calculates their total deflection field
splitting the work load in a number of threads. Using mpi4py.
To run the code:

mpiexec -n 6 python test_lensparallel.py


NOTE: this code differs from test_lensparallel.py because the lenses are modeled using lenstronomy instead
of pyLensLib

"""
import numpy as np
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from pyLensLib.sie import sie
import matplotlib.pyplot as plt
import time


# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t0 = time.time()

# create an instance of a lens, for example a piemd model:

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

nlens = 1000

sz = 100
n  = 1000
theta=np.linspace(-sz/2.,sz/2.,n)
theta1, theta2 = np.meshgrid(theta, theta)

shiftx = np.empty(nlens)
shifty = np.empty(nlens)
if rank == 0:
    shiftx = np.random.random(nlens)*sz-sz/2.
    shifty = np.random.random(nlens)*sz-sz/2.

comm.Bcast(shiftx, root=0)
comm.Bcast(shifty, root=0)

dpie_list = []
dpie_list_kw = []

kwargs = {'zl': 0.5,
          'zs': 2.0,
          'sigma0': 1000.0,
          'q': 1.0,
          'pa': -np.pi / 4.0,
          'x1': 0.0,
          'x2': 0.0}

si_ref = sie(co,**kwargs)

for i in range(nlens):

    kwargs = {
          'theta_E': si_ref.bsie(),
          'e1': 0.0,
          'e2': 0.0,
          'center_x': 0.0+shiftx[i],
          'center_y': 0.0+shifty[i]}
    dpie_list_kw.append(kwargs)
    dpie_list.append('SIE')

lens_model = LensModel(dpie_list,z_lens=0.5,z_source=1.0,cosmo=co,NIE=False)


# Compute the number of items to send to each process
counts = np.full(size, n*n // size)
counts[:n*n % size] += 1

if rank == 0:
    print (size,counts)

displs = np.concatenate(([0], np.cumsum(counts[:-1])))

# Scatter the grid of (x,y) coordinates
x_recvbuf = np.empty(counts[rank])
y_recvbuf = np.empty(counts[rank])

comm.Scatterv([theta1.ravel(), counts, displs, MPI.DOUBLE], x_recvbuf, root=0)
comm.Scatterv([theta2.ravel(), counts, displs, MPI.DOUBLE], y_recvbuf, root=0)

# Compute deflection angle for this part of the grid
alpha_x, alpha_y = lens_model.alpha(x_recvbuf, y_recvbuf,dpie_list_kw)


# Gather the deflection angles
alpha_x = comm.gather(alpha_x, root=0)
alpha_y = comm.gather(alpha_y, root=0)

# Assemble the full deflection angle map
if rank == 0:
    alpha_x = np.concatenate(alpha_x)
    alpha_y = np.concatenate(alpha_y)
    alpha_x = alpha_x.reshape((n, n))
    alpha_y = alpha_y.reshape((n, n))
    print('done, time = ', time.time() - t0)
    fig, ax = plt.subplots(1,3,figsize=(18,10))
    ax[0].imshow(alpha_x)
    ax[1].imshow(alpha_y)
    a21,a11=np.gradient(alpha_x)
    a22,a12=np.gradient(alpha_y)
    ax[2].imshow(0.5*(a11+a22))
    plt.show()
