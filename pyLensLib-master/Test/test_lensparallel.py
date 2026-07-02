"""
This code generates nlens lenses and calculates their total deflection field
splitting the work load in a number of threads. Using mpi4py.
To run the code:

mpiexec -n 6 python test_lensparallel.py

"""
import numpy as np
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
#from pyLensLib.piemd import piemd
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
for i in range(nlens):
    kwargs = {'zl': 0.5,
          'zs': 2.0,
          'sigma0': 1000.0,
          'q': 1.0,
          'pa': -np.pi / 4.0,
          'theta_c': 0.0,
          'theta_t': 1000.0,
          'x1': 0.0+shiftx[i],
          'x2': 0.0+shifty[i]}
    dpie_list.append(sie(co, **kwargs))



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
for i in range(nlens):
    alpha_x_, alpha_y_ = dpie_list[i].angle(x_recvbuf, y_recvbuf)
    if i == 0:
        alpha_x = alpha_x_
        alpha_y = alpha_y_
    else:
        alpha_x = alpha_x + alpha_x_
        alpha_y = alpha_y + alpha_y_


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
