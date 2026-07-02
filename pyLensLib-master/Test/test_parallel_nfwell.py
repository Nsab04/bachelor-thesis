# script to run the construction of multiple NFW halos with different masses in parallel and compute 
# the total convergence map

import numpy as np
import time
import multiprocessing as mp
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
from pyLensLib.nfwell import nfwell
import argparse
from pyLensLib.deflector import deflector


def main_kappa(co, zl, zs, n, ncores, npix, xmin=-1, xmax=1, ymin=-1, ymax=1):
    # set the masses of the NFW halos
    mass = np.logspace(14, 15, n)

    # set random positions between -1 and 1 Mpc in the x and y directions
    x = np.random.uniform(xmin, xmax, n)
    y = np.random.uniform(ymin, ymax, n)
    
    # assume that only two cores are available for the computation
    # divide the computation among the available cores

    print ('Number of cores: ', ncores)
    print ('Number of halos: ', n)
    # create a pool of 2 processes
    pool = mp.Pool(processes=ncores)
    # compute the convergence for each halo and make the sum
    try:
        kappa = pool.starmap(kappaNFW, zip([co]*n, mass, x, y, [zl]*n, [zs]*n, [npix]*n))
    finally:
        pool.close()
        pool.join()
    return np.sum(kappa, axis=0)


def main_alpha(co, zl, zs, n, ncores, npix, xmin=-1, xmax=1, ymin=-1, ymax=1):

    # set the masses of the NFW halos
    mass = np.logspace(12, 13, n)
    # append a main halo to the mass array
    mass = np.append(mass,1e15)

    # set random positions between -1 and 1 Mpc in the x and y directions
    x = np.random.uniform(xmin, xmax, n)
    y = np.random.uniform(ymin, ymax, n)
    x = np.append(x,0.0)
    y = np.append(y,0.0)


    # assume that only two cores are available for the computation
    # divide the computation among the available cores

    n+=1
    print('Number of cores: ', ncores)
    print('Number of halos: ', n)
    # create a pool of 2 processes
    pool = mp.Pool(processes=ncores)
    # compute the convergence for each halo and make the sum
    try:
        results = pool.starmap(anglesNFW, zip([co] * n, mass,
                                              x, y, [zl] * n, [zs] * n, [npix] * n,
                                              [xmin] * n, [xmax] * n, [ymin] * n, [ymax] * n))
        a1,a2 = zip(*results)
    finally:
        pool.close()
        pool.join()
    return np.sum(a1, axis=0), np.sum(a2,axis=0)
    

# function to compute the convergence for a single NFW halo
def kappaNFW(co, mass, x, y, zl, zs, npix, xmin, xmax, ymin, ymax):
    kwargs =  {'zl': zl, 'zs': zs, 'mass': mass, 'conc': 4.0, 'x1': x, 'x2': y}
    ne = nfwell(co,**kwargs)
    theta1 = np.linspace(xmin,xmax,npix)
    theta2 = np.linspace(ymin, ymax, npix)
    ne.setGrid(theta1,theta2)
    ka = ne.ka # compute the convergence
    return ka

def anglesNFW(co,mass,x,y,zl,zs,npix, xmin, xmax, ymin, ymax):
    kwargs = {'zl': zl, 'zs': zs, 'mass': mass, 'conc': 4.0, 'x1': x, 'x2': y}
    ne = nfwell(co, **kwargs)
    theta1 = np.linspace(xmin,xmax,npix)
    theta2 = np.linspace(ymin, ymax, npix)
    thetax,thetay = np.meshgrid(theta1,theta2)
    a1, a2 = ne.angle(thetax,thetay)
    return a1, a2

    
# compute the total convergence map for the 6 NFW halos in parallel. The total convergence is 
# the sum of the convergence of the individual halos



if __name__ == "__main__":

    # parse the command line arguments: the number of halos, the number of cores, and the number of pixels

    parser = argparse.ArgumentParser(description='Compute the total convergence map for a set of NFW halos in parallel')


    parser.add_argument('-n', '--n_halos', required=True, type=int, help='Number of NFW halos')
    parser.add_argument('-c', '--n_cores', required=True, type=int, help='Number of cores')
    parser.add_argument('-p', '--n_pix', required=True, type=int, help='Number of pixels')
    zl = 0.5
    zs = 1.0
    co = FlatLambdaCDM(H0=70.0, Om0=0.3)
    xmin = -20
    ymin = -20
    xmax = 20
    ymax = 20

    args = parser.parse_args()
    n = args.n_halos
    ncores = args.n_cores
    npix = args.n_pix


    # time the execution of the main function
    start = time.time()
    a1, a2 = main_alpha(co, zl, zs, n, ncores, npix, xmin, xmax, ymin, ymax)
    end = time.time()
    
    print ('Execution time: ', end-start)


    kwargs_def = {'zl': zl, 'zs': zs}
    df = deflector(co,angx=a1,angy=a2,**kwargs_def)
    theta = np.linspace(xmin, xmax, npix)
    df.setGrid(theta, theta)
    plt,ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(df.ka, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='viridis')
    tl = df.tancl()
    for t in tl:
        x,y = df.getCritPoints(t)
        ax.plot(x,y,'--',color='white')
    tl = df.radcl()
    for t in tl:
        x,y = df.getCritPoints(t)
        ax.plot(x,y,'--',color='white')
    plt.show()
    print ('done')
    # wait for the figure to be closed
    plt.waitforbuttonpress()
    




    