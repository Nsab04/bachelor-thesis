import numpy as np
from scipy.interpolate import interp1d

def ellipsoidal_distribution(n=100,saxes=[1.0,0.5]):
    """
    Uniformly distribute points within an ellipsoidal volume.

    Args:
        n (int): Number of points to generate.
        saxes (array): Semi-axes of the ellipsoidal volume.

    Returns:
        ndarray: Coordinates of distributed points.
    """
    # sort the semi-axes:
    saxes=np.sort(np.atleast_1d(saxes))[::-1]
    naxes = len(saxes)

    if naxes <=1:
        raise ValueError('dimensions must be >=2')

    x = np.random.normal(size=(n, naxes), scale=saxes)
    r = np.sqrt(np.sum((x / saxes) ** 2, axis=-1))
    rs = (np.random.random(n)) ** (1.0 / naxes)

    r = r/rs
    return (1.0/r[:,np.newaxis])*x


def sampleLensingDistance(co, zl, zsmax=6.0, nplanes=20):
    """
    Create a list of source planes in the background of a lens, sampling uniformly the lensing distance growth.

    Args:
        co: Cosmology instance.
        zl (float): Lens redshift.
        zsmax (float): Maximum source redshift.
        nplanes (int): Number of source planes.

    Returns:
        ndarray: List of source plane redshifts.
    """
    dl = co.angular_diameter_distance(zl)
    z = np.linspace(zl + 0.1, zsmax, 100)
    dls = co.angular_diameter_distance_z1z2(zl, z)
    ds = co.angular_diameter_distance(z)
    dlens = (dl * dls / ds).value
    f = interp1d(dlens, z)
    dlens_ = np.linspace(dlens.min(), dlens.max(), nplanes)
    z_ = f(dlens_)
    return z_

def sample2Dimage(image,n=1000):
    """
    Generate n points on an image, distributed according to image intensity.

    Args:
        image (ndarray): Input image.
        n (int): Number of points to generate.

    Returns:
        tuple: Arrays of x and y coordinates of sampled points.
    """
    values = image.flatten()
    x_ = np.linspace(0, image.shape[0]-1 ,image.shape[0])
    y_ = np.linspace(0, image.shape[1]-1, image.shape[1])
    x, y = np.meshgrid(x_,y_,indexing='ij')
    xx = x.flatten()
    yy = y.flatten()
    ind = np.indices([len(values)])#np.arange(0,len(values)-1,1)
    sample = np.random.choice(ind[0],n,p=values/np.sum(values))
    dx = np.random.rand(n)-0.5
    dy = np.random.rand(n)-0.5
    return xx[sample]+dx,yy[sample]+dy

def sample3Dcube(cube,n=1000):
    """
    Generate n points in a 3D cube, distributed according to cube intensity.

    Args:
        cube (ndarray): Input 3D array.
        n (int): Number of points to generate.

    Returns:
        tuple: Arrays of x, y, z coordinates of sampled points.
    """
    values = cube.flatten()

    x, y, z = np.mgrid[0:cube.shape[0], 0:cube.shape[1], 0:cube.shape[2]]

    xx = x.flatten()
    yy = y.flatten()
    zz = z.flatten()

    ind = np.indices([len(values)])#np.arange(0,len(values)-1,1)
    sample = np.random.choice(ind[0],n,p=values/np.sum(values))
    dx = np.random.rand(n) - 0.5
    dy = np.random.rand(n) - 0.5
    dz = np.random.rand(n) - 0.5
    return xx[sample]+dx, yy[sample]+dy, zz[sample]+dz


def PowerLawExpCut(x,xcut=0.8,beta=-0.9,gamma=3.0,delta=1.0):
    """
    Compute a power-law with exponential cutoff distribution.

    Args:
        x (ndarray): Input values.
        xcut (float): Cutoff value.
        beta (float): Power-law exponent.
        gamma (float): Exponential cutoff exponent.
        delta (float): Exponential cutoff scale.

    Returns:
        ndarray: Normalized distribution values.
    """
    f = x**beta*np.exp(-delta*(x/xcut)**gamma)
    f = f/np.sum(f)
    return f

def samplePowerLawExCut(xtot=1.0,fsub=0.1,
                        xmin=1e-6,xmax=1.0,xcut=0.8,nx=10000,
                        beta=-0.9,gamma=3.0,delta=1.0):
    """
    Sample values from a power-law with exponential cutoff until a fraction fsub of xtot is reached.

    Args:
        xtot (float): Total value to reach.
        fsub (float): Fractional threshold.
        xmin (float): Minimum value.
        xmax (float): Maximum value.
        xcut (float): Cutoff value.
        nx (int): Number of samples.
        beta (float): Power-law exponent.
        gamma (float): Exponential cutoff exponent.
        delta (float): Exponential cutoff scale.

    Returns:
        list: Sampled values.
    """

    x = np.linspace(xmin, xmax, nx)
    ind = np.indices([len(x)])
    p = PowerLawExpCut(x,xcut=xcut,beta=beta,gamma=gamma,delta=delta)

    fsub_=0.0
    x_=[]
    xsum = 0.0
    while fsub_ < fsub:
        sample = np.random.choice(ind[0],1,p=p)
        x_.append(x[sample][0])
        xsum+=x[sample][0]
        fsub_ = xsum/xtot

    return x_

def sample_concentration(M=1e15, A=5.0, B=-0.1, M0=1e14, scatter=0.2):
    """
    Sample the concentration for a galaxy cluster of mass M using a log-normal scatter.

    Args:
        M (float): Mass of the galaxy cluster.
        A (float): Normalization constant in the c-M relation.
        B (float): Exponent in the c-M relation.
        M0 (float): Normalization mass in the c-M relation.
        scatter (float): Standard deviation of the log-normal scatter in the c-M relation.

    Returns:
        float: Sampled concentration for the galaxy cluster.
    """
    # Calculate the mean concentration using the c-M relation
    mean_concentration = A * (M / M0)**B

    # Sample from a log-normal distribution to account for scatter
    log_mean = np.log(mean_concentration)
    concentration = np.random.lognormal(mean=log_mean, sigma=scatter)

    return concentration

def assign_intrinsic_ellipticity(sigma):
    """
    Assign an intrinsic ellipticity to a mock galaxy using a Rayleigh distribution.

    Args:
        sigma (float): Scale parameter of the Rayleigh distribution.

    Returns:
        float: Assigned intrinsic ellipticity.
    """
    ellipticity = np.random.rayleigh(sigma)
    return ellipticity




