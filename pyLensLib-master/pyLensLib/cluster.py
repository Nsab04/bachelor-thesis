#import pyLensLib.gadget  as G
import g3read
import numpy as np
import scipy.ndimage as ndimage
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
import random
import h5py
import warnings

try:
    from sphviewer.tools import QuickView
    _SPHVIEWER_AVAILABLE = True
    _SPHVIEWER_BACKEND = "sphviewer"
except Exception:  # pragma: no cover - optional dependency
    QuickView = None
    try:
        import sphviewer2 as _sphviewer2
        _SPHVIEWER_AVAILABLE = True
        _SPHVIEWER_BACKEND = "sphviewer2"
    except Exception:
        _sphviewer2 = None
        _SPHVIEWER_AVAILABLE = False
        _SPHVIEWER_BACKEND = "internal"

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # pragma: no cover - fallback path
        def _wrap(func):
            return func
        return _wrap


import sys, math

_SPH_KERNEL_CUBIC_SPLINE = 0
_SPH_KERNEL_WENDLAND_C2 = 1
_SPH_KERNEL_CUBIC_SPLINE_GAMMA = 2.0
_SPH_KERNEL_WENDLAND_C2_GAMMA = 1.936492
_SPH_KERNEL_IDS = {
    "cubic_spline": _SPH_KERNEL_CUBIC_SPLINE,
    "cubic-spline": _SPH_KERNEL_CUBIC_SPLINE,
    "cubic": _SPH_KERNEL_CUBIC_SPLINE,
    "wendland_c2": _SPH_KERNEL_WENDLAND_C2,
    "wendland-c2": _SPH_KERNEL_WENDLAND_C2,
    "wendland": _SPH_KERNEL_WENDLAND_C2,
}


def _sph_kernel_id(kernel):
    """
    Return the internal numeric identifier for a supported SPH deposition kernel.
    """
    try:
        key = str(kernel).strip().lower()
    except Exception as exc:
        raise ValueError("SPH kernel name must be a string.") from exc
    if key not in _SPH_KERNEL_IDS:
        valid = "cubic_spline, wendland_c2"
        raise ValueError(f"Unsupported SPH kernel {kernel!r}. Valid kernels are: {valid}.")
    return _SPH_KERNEL_IDS[key]


def _sph_kernel_gamma(kernel_id, kernel_gamma=None):
    """
    Return the compact-support radius, in units of smoothing length, for a kernel.
    """
    if kernel_id == _SPH_KERNEL_WENDLAND_C2:
        gamma = _SPH_KERNEL_WENDLAND_C2_GAMMA if kernel_gamma is None else float(kernel_gamma)
        if gamma <= 0.0:
            raise ValueError("Wendland-C2 kernel_gamma must be positive.")
        return gamma
    return _SPH_KERNEL_CUBIC_SPLINE_GAMMA


def _compute_hsml_knn(pos, nb=32, min_hsml=1e-5, max_hsml=0.1):
    """
    Estimate SPH smoothing lengths from k-nearest-neighbor distances.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    nb : int
        Number of neighbors used for hsml estimation.
    min_hsml, max_hsml : float
        Bounds applied to hsml values.

    Returns
    -------
    ndarray, shape (N,)
        Smoothing lengths.
    """
    n = pos.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.array([float(np.clip(0.5 * (min_hsml + max_hsml), min_hsml, max_hsml))], dtype=np.float64)

    k = int(max(2, min(nb + 1, n)))
    tree = cKDTree(pos)
    try:
        dists, _ = tree.query(pos, k=k, workers=-1)
    except TypeError:
        dists, _ = tree.query(pos, k=k)

    if dists.ndim == 1:
        hsml = dists
    else:
        hsml = dists[:, -1]
    hsml = np.clip(hsml, min_hsml, max_hsml)
    return hsml.astype(np.float64, copy=False)


@njit(cache=False, fastmath=True)
def _kernel_cubic_spline_2d(q, h):
    """
    2D cubic-spline SPH kernel with compact support q < 2.
    """
    if q >= 2.0 or h <= 0.0:
        return 0.0
    sigma = 10.0 / (7.0 * np.pi * h * h)
    if q < 1.0:
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    t = 2.0 - q
    return sigma * (0.25 * t * t * t)


def _kernel_cubic_spline_2d_py(q, h):
    """
    Python implementation of the 2D cubic-spline SPH kernel.
    """
    if q >= 2.0 or h <= 0.0:
        return 0.0
    sigma = 10.0 / (7.0 * np.pi * h * h)
    if q < 1.0:
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    t = 2.0 - q
    return sigma * (0.25 * t * t * t)


@njit(cache=False, fastmath=True)
def _kernel_wendland_c2_2d(q, h, kernel_gamma):
    """
    2D Wendland-C2 SPH kernel with compact support q < kernel_gamma.
    """
    if q >= kernel_gamma or h <= 0.0 or kernel_gamma <= 0.0:
        return 0.0
    u = q / kernel_gamma
    t = 1.0 - u
    return (7.0 / (np.pi * kernel_gamma * kernel_gamma * h * h)) * t * t * t * t * (1.0 + 4.0 * u)


def _kernel_wendland_c2_2d_py(q, h, kernel_gamma=_SPH_KERNEL_WENDLAND_C2_GAMMA):
    """
    Python implementation of the 2D Wendland-C2 SPH kernel.
    """
    if q >= kernel_gamma or h <= 0.0 or kernel_gamma <= 0.0:
        return 0.0
    u = q / kernel_gamma
    t = 1.0 - u
    return (7.0 / (np.pi * kernel_gamma * kernel_gamma * h * h)) * t * t * t * t * (1.0 + 4.0 * u)


@njit(cache=False, fastmath=True)
def _kernel_sph_2d(q, h, kernel_id, kernel_gamma):
    if kernel_id == _SPH_KERNEL_WENDLAND_C2:
        return _kernel_wendland_c2_2d(q, h, kernel_gamma)
    return _kernel_cubic_spline_2d(q, h)


def _kernel_sph_2d_py(q, h, kernel_id, kernel_gamma):
    if kernel_id == _SPH_KERNEL_WENDLAND_C2:
        return _kernel_wendland_c2_2d_py(q, h, kernel_gamma)
    return _kernel_cubic_spline_2d_py(q, h)


@njit(cache=False, fastmath=True)
def _deposit_sph_mass_numba(
    x, y, mass, hsml, img, x_centers, y_centers, xmin, xmax, ymin, ymax,
    kernel_id, kernel_gamma
):
    """
    Deposit particle masses to a 2D grid using a compact SPH kernel.
    """
    ny, nx = img.shape
    inv_dx = nx / (xmax - xmin)
    inv_dy = ny / (ymax - ymin)

    for i in range(x.size):
        hi = hsml[i]
        if hi <= 0.0:
            continue

        hi2 = hi * hi
        inv_hi2 = 1.0 / hi2
        cutoff2 = kernel_gamma * kernel_gamma * hi2
        xi = x[i]
        yi = y[i]
        ri = kernel_gamma * hi

        x0 = int(np.floor((xi - ri - xmin) * inv_dx))
        x1 = int(np.floor((xi + ri - xmin) * inv_dx))
        y0 = int(np.floor((yi - ri - ymin) * inv_dy))
        y1 = int(np.floor((yi + ri - ymin) * inv_dy))

        if x1 < 0 or x0 >= nx or y1 < 0 or y0 >= ny:
            continue

        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 >= nx:
            x1 = nx - 1
        if y1 >= ny:
            y1 = ny - 1

        wsum = 0.0
        for iy in range(y0, y1 + 1):
            dy = y_centers[iy] - yi
            dy2 = dy * dy
            for ix in range(x0, x1 + 1):
                dx = x_centers[ix] - xi
                r2 = dx * dx + dy2
                if r2 >= cutoff2:
                    continue
                q2 = r2 * inv_hi2
                q = np.sqrt(q2)
                w = _kernel_sph_2d(q, hi, kernel_id, kernel_gamma)
                if w > 0.0:
                    wsum += w

        if wsum <= 0.0:
            ix = int(np.floor((xi - xmin) * inv_dx))
            iy = int(np.floor((yi - ymin) * inv_dy))
            if 0 <= ix < nx and 0 <= iy < ny:
                img[iy, ix] += mass[i]
            continue

        norm = mass[i] / wsum
        for iy in range(y0, y1 + 1):
            dy = y_centers[iy] - yi
            dy2 = dy * dy
            for ix in range(x0, x1 + 1):
                dx = x_centers[ix] - xi
                r2 = dx * dx + dy2
                if r2 >= cutoff2:
                    continue
                q2 = r2 * inv_hi2
                q = np.sqrt(q2)
                w = _kernel_sph_2d(q, hi, kernel_id, kernel_gamma)
                if w > 0.0:
                    img[iy, ix] += norm * w


@njit(cache=False, fastmath=True)
def _deposit_sph_mass_numba_fast(
    x, y, mass, hsml, img, x_centers, y_centers, xmin, xmax, ymin, ymax,
    kernel_id, kernel_gamma
):
    """
    Fast one-pass deposition: accumulate m_i * W(r, h_i) and renormalize globally.
    """
    ny, nx = img.shape
    inv_dx = nx / (xmax - xmin)
    inv_dy = ny / (ymax - ymin)

    for i in range(x.size):
        hi = hsml[i]
        if hi <= 0.0:
            continue

        hi2 = hi * hi
        inv_hi2 = 1.0 / hi2
        cutoff2 = kernel_gamma * kernel_gamma * hi2
        xi = x[i]
        yi = y[i]
        ri = kernel_gamma * hi

        x0 = int(np.floor((xi - ri - xmin) * inv_dx))
        x1 = int(np.floor((xi + ri - xmin) * inv_dx))
        y0 = int(np.floor((yi - ri - ymin) * inv_dy))
        y1 = int(np.floor((yi + ri - ymin) * inv_dy))

        if x1 < 0 or x0 >= nx or y1 < 0 or y0 >= ny:
            continue

        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 >= nx:
            x1 = nx - 1
        if y1 >= ny:
            y1 = ny - 1

        mi = mass[i]
        for iy in range(y0, y1 + 1):
            dy = y_centers[iy] - yi
            dy2 = dy * dy
            for ix in range(x0, x1 + 1):
                dx = x_centers[ix] - xi
                r2 = dx * dx + dy2
                if r2 >= cutoff2:
                    continue
                q2 = r2 * inv_hi2
                q = np.sqrt(q2)
                w = _kernel_sph_2d(q, hi, kernel_id, kernel_gamma)
                if w > 0.0:
                    img[iy, ix] += mi * w


def _deposit_sph_mass_python(
    x, y, mass, hsml, img, x_centers, y_centers, xmin, xmax, ymin, ymax,
    kernel_id, kernel_gamma
):
    """
    Python fallback for SPH deposition when numba is unavailable.
    """
    ny, nx = img.shape
    inv_dx = nx / (xmax - xmin)
    inv_dy = ny / (ymax - ymin)

    for i in range(x.size):
        hi = hsml[i]
        if hi <= 0.0:
            continue
        xi = x[i]
        yi = y[i]
        ri = kernel_gamma * hi

        x0 = int(np.floor((xi - ri - xmin) * inv_dx))
        x1 = int(np.floor((xi + ri - xmin) * inv_dx))
        y0 = int(np.floor((yi - ri - ymin) * inv_dy))
        y1 = int(np.floor((yi + ri - ymin) * inv_dy))

        if x1 < 0 or x0 >= nx or y1 < 0 or y0 >= ny:
            continue
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(nx - 1, x1)
        y1 = min(ny - 1, y1)

        wsum = 0.0
        for iy in range(y0, y1 + 1):
            dy = y_centers[iy] - yi
            for ix in range(x0, x1 + 1):
                dx = x_centers[ix] - xi
                q = np.sqrt(dx * dx + dy * dy) / hi
                wsum += _kernel_sph_2d_py(q, hi, kernel_id, kernel_gamma)

        if wsum <= 0.0:
            ix = int(np.floor((xi - xmin) * inv_dx))
            iy = int(np.floor((yi - ymin) * inv_dy))
            if 0 <= ix < nx and 0 <= iy < ny:
                img[iy, ix] += mass[i]
            continue

        norm = mass[i] / wsum
        for iy in range(y0, y1 + 1):
            dy = y_centers[iy] - yi
            for ix in range(x0, x1 + 1):
                dx = x_centers[ix] - xi
                q = np.sqrt(dx * dx + dy * dy) / hi
                w = _kernel_sph_2d_py(q, hi, kernel_id, kernel_gamma)
                if w > 0.0:
                    img[iy, ix] += norm * w


def _deposit_sph_mass_python_fast(
    x, y, mass, hsml, img, x_centers, y_centers, xmin, xmax, ymin, ymax,
    kernel_id, kernel_gamma
):
    """
    Python fallback for fast one-pass deposition.
    """
    ny, nx = img.shape
    inv_dx = nx / (xmax - xmin)
    inv_dy = ny / (ymax - ymin)

    for i in range(x.size):
        hi = hsml[i]
        if hi <= 0.0:
            continue
        xi = x[i]
        yi = y[i]
        ri = kernel_gamma * hi

        x0 = int(np.floor((xi - ri - xmin) * inv_dx))
        x1 = int(np.floor((xi + ri - xmin) * inv_dx))
        y0 = int(np.floor((yi - ri - ymin) * inv_dy))
        y1 = int(np.floor((yi + ri - ymin) * inv_dy))

        if x1 < 0 or x0 >= nx or y1 < 0 or y0 >= ny:
            continue
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(nx - 1, x1)
        y1 = min(ny - 1, y1)

        mi = mass[i]
        for iy in range(y0, y1 + 1):
            dy = y_centers[iy] - yi
            for ix in range(x0, x1 + 1):
                dx = x_centers[ix] - xi
                q = np.sqrt(dx * dx + dy * dy) / hi
                w = _kernel_sph_2d_py(q, hi, kernel_id, kernel_gamma)
                if w > 0.0:
                    img[iy, ix] += mi * w


def _mass_map_sph_native(
    pos, mass, npix=512, xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0,
    nb=32, min_hsml=1e-5, max_hsml=0.1, use_numba=True,
    local_mass_conservation=False, kernel="cubic_spline", kernel_gamma=None
):
    """
    Build an SPH-projected mass map without external py-sphviewer dependency.
    """
    kernel_id = _sph_kernel_id(kernel)
    kernel_gamma = _sph_kernel_gamma(kernel_id, kernel_gamma=kernel_gamma)
    img = np.zeros((npix, npix), dtype=np.float64)
    if mass.size == 0:
        return img

    hsml = _compute_hsml_knn(
        pos=pos, nb=nb, min_hsml=min_hsml, max_hsml=max_hsml
    )
    x = np.asarray(pos[:, 0], dtype=np.float64)
    y = np.asarray(pos[:, 1], dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)

    x_centers = xmin + (np.arange(npix, dtype=np.float64) + 0.5) * (xmax - xmin) / npix
    y_centers = ymin + (np.arange(npix, dtype=np.float64) + 0.5) * (ymax - ymin) / npix

    if use_numba and _NUMBA_AVAILABLE:
        if local_mass_conservation:
            _deposit_sph_mass_numba(
                x, y, m, hsml, img, x_centers, y_centers, xmin, xmax, ymin, ymax,
                kernel_id, kernel_gamma
            )
        else:
            _deposit_sph_mass_numba_fast(
                x, y, m, hsml, img, x_centers, y_centers, xmin, xmax, ymin, ymax,
                kernel_id, kernel_gamma
            )
    else:
        if local_mass_conservation:
            _deposit_sph_mass_python(
                x, y, m, hsml, img, x_centers, y_centers, xmin, xmax, ymin, ymax,
                kernel_id, kernel_gamma
            )
        else:
            _deposit_sph_mass_python_fast(
                x, y, m, hsml, img, x_centers, y_centers, xmin, xmax, ymin, ymax,
                kernel_id, kernel_gamma
            )

    total_img_mass = img.sum()
    total_particles_mass = m.sum()
    if total_img_mass > 0.0 and total_particles_mass > 0.0:
        img = img / total_img_mass * total_particles_mass
    return img


def _resize_mass_map(img, npix):
    """
    Resize a square mass map to the requested size and preserve total mass.
    """
    if img.shape == (npix, npix):
        return img
    total = img.sum()
    zoom = (float(npix) / img.shape[0], float(npix) / img.shape[1])
    resized = ndimage.zoom(img, zoom, order=1)
    if resized.shape != (npix, npix):
        resized = resized[:npix, :npix]
        if resized.shape[0] < npix or resized.shape[1] < npix:
            padded = np.zeros((npix, npix), dtype=resized.dtype)
            padded[:resized.shape[0], :resized.shape[1]] = resized
            resized = padded
    resized_total = resized.sum()
    if total > 0.0 and resized_total > 0.0:
        resized = resized / resized_total * total
    return resized


def _mass_map_sphviewer2(
    pos, mass, npix=512, xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0,
    zmin=-3.0, zmax=3.0, nb=32, min_hsml=1e-5, max_hsml=0.1
):
    """
    Build an SPH-projected mass map with the py-sphviewer2 API.

    py-sphviewer2 does not expose the old QuickView interface. It expects
    smoothing lengths explicitly, so we reuse pyLensLib's kNN estimator and pass
    the requested mass-map bounds through a non-periodic camera.
    """
    img = np.zeros((npix, npix), dtype=np.float64)
    if mass.size == 0:
        return img

    xwidth = float(xmax - xmin)
    ywidth = float(ymax - ymin)
    zwidth = float(zmax - zmin)
    if xwidth <= 0.0 or ywidth <= 0.0 or zwidth <= 0.0:
        raise ValueError("massMapSPH bounds must have positive width along x, y, and z.")

    render_npix = int(2 ** np.ceil(np.log2(max(1, npix))))
    r_max = int(np.log2(render_npix))
    extent = max(xwidth, ywidth)
    hsml = _compute_hsml_knn(
        pos=pos, nb=nb, min_hsml=min_hsml, max_hsml=max_hsml
    )

    image, image_extent = _sphviewer2.render(
        pos[:, 0], pos[:, 1], pos[:, 2], hsml, mass,
        Lbox=max(extent, zwidth, 4.0 * max_hsml, np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), np.ptp(pos[:, 2])),
        extent=extent,
        xc=0.5 * (xmin + xmax),
        yc=0.5 * (ymin + ymax),
        zc=0.5 * (zmin + zmax),
        periodic=False,
        r_max=r_max,
        target_cells_per_h=4,
    )
    image = np.asarray(image, dtype=np.float64)

    if xwidth != extent or ywidth != extent:
        x0, x1, y0, y1 = image_extent
        nx = image.shape[1]
        ny = image.shape[0]
        ix0 = max(0, int(np.floor((xmin - x0) / (x1 - x0) * nx)))
        ix1 = min(nx, int(np.ceil((xmax - x0) / (x1 - x0) * nx)))
        iy0 = max(0, int(np.floor((ymin - y0) / (y1 - y0) * ny)))
        iy1 = min(ny, int(np.ceil((ymax - y0) / (y1 - y0) * ny)))
        image = image[iy0:iy1, ix0:ix1]
        if image.size == 0:
            return img

    img = _resize_mass_map(image, npix)
    total_img_mass = img.sum()
    total_particles_mass = mass.sum()
    if total_img_mass > 0.0 and total_particles_mass > 0.0:
        img = img / total_img_mass * total_particles_mass
    return img


class cluster(object):
    """
    Represents a galaxy cluster created from a distribution of particles in a gadget snapshot file.

    Attributes:
        snapshot (str): Name of the gadget snapshot file.
        parttype (list): List of particle types to use.
        xc, yc, zc (float): Coordinates of the halo center.
        sn (str): Snapshot number.
        cln (str): Halo number in the snapshot.
        IT (np.ndarray): Matrix to change coordinate system.
        fsample (float): Fraction of particles to randomly sample.
        readpot (bool): Whether to read particle potentials.
        alignIT (bool): Whether to switch to the coordinate system defined by IT.
        hdf5 (bool): Whether to read snapshot in hdf5 format.
        verbose (bool): Verbosity flag.
    """
    
    def __init__(self, snapshot, parttype, xc=0.0, yc=0.0, zc=0.0, sn='0', cln='0',
                 IT=np.identity(3), fsample=0.0, readpot=False, alignIT=False, hdf5=False, verbose=False):
        """
        Initialize a Cluster instance.

        Args:
            snapshot (str): Name of the gadget snapshot file.
            parttype (list): List of particle types to use (e.g., [0,1,2,3,4]).
            xc (float): X coordinate of the halo center.
            yc (float): Y coordinate of the halo center.
            zc (float): Z coordinate of the halo center.
            sn (str): Snapshot number.
            cln (str): Halo number in the snapshot.
            IT (np.ndarray): Matrix to change coordinate system.
            fsample (float): Fraction of particles to randomly sample.
            readpot (bool): Whether to read particle potentials.
            alignIT (bool): Whether to switch to the coordinate system defined by IT.
            hdf5 (bool): Whether to read snapshot in hdf5 format.
            verbose (bool): Verbosity flag.
        """
        self.snapshot = snapshot
        self.parttype = parttype
        self.verbose = verbose
        mass=np.array([])
        ptype=np.array([])
        pid=np.array([])

        pos= np.array([]).reshape(0,3)
        if hdf5:
            sim_file = h5py.File(snapshot, 'r')
            for i in range(len(parttype)):
                partType_ = 'PartType' + str(parttype[i])
                pos_tmp = sim_file[partType_+'/Coordinates'][:]
                mass_tmp = sim_file[partType_+'/Masses'][:]
                ptype_tmp = np.ones(mass_tmp.size) * parttype[i]
                pid_tmp = sim_file[partType_+'/ParticleIDs'][:]
                mass = np.append(mass, mass_tmp)
                pos = np.append(pos, pos_tmp, axis=0)
                ptype = np.append(ptype, ptype_tmp)
                pid = np.append(pid, pid_tmp, axis=0)
        else:
            for i in range(len(parttype)):
                """
                mass_tmp=G.read_block(self.snapshot,"MASS",parttype=parttype[i])
                ptype_tmp=np.ones(mass_tmp.size)*parttype[i]
                mass=np.append(mass,mass_tmp)
                ptype=np.append(ptype,ptype_tmp)
                pos_tmp=G.read_block(self.snapshot,"POS",parttype=parttype[i])
                pos=np.append(pos,pos_tmp,axis=0)
                pid_tmp=G.read_block(self.snapshot,"ID",parttype=parttype[i])
                pid=np.append(pid,pid_tmp,axis=0)
                if self.verbose:
                    print ('Particle type, number:',parttype[i],len(mass_tmp))
                """
                data = g3read.read_new(self.snapshot, ["POS ", "MASS", "ID  "], parttype[i])
                #posi  = g3read.read_new(self.snapshot, "POS ", parttype[i])
                #massi = g3read.read_new(self.snapshot, "MASS", parttype[i])
                #pidi = g3read.read_new(self.snapshot, "ID  ", parttype[i])
                #print('posi', posi.shape)
                pos = np.append(pos,data["POS "],axis=0)
                mass = np.append(mass,data["MASS"],axis=0)
                pid = np.append(pid,data["ID  "],axis=0)
                ptype_tmp = np.ones(data["MASS"].size) * parttype[i]
                ptype = np.append(ptype, ptype_tmp)
        if readpot:
            pot=np.array([])
            if hdf5:
                for i in range(len(parttype)):
                    partType_ = 'PartType' + str(parttype[i])
                    pot_tmp = sim_file[partType_+'/Potential'][:]
                    pot = np.append(pot, pot_tmp, axis=0)
            else:
                for i in range(len(parttype)):
                    """
                    pot_tmp=G.read_block(self.snapshot,"POT",parttype=parttype[i])
                    """
                    pot_tmp = g3read.read_new(self.snapshot,"POT ",parttype[i])
                    pot=np.append(pot,pot_tmp,axis=0)

        if fsample > 0.0:
            npart = len(mass)
            if fsample <= 1.0:
                # fsample is interpreted as the sampled fraction of particles.
                nsample = int(max(1, round(npart * fsample)))
                mass_weight = 1.0 / fsample
            else:
                # Backward compatibility: fsample > 1 acts as downsampling factor.
                nsample = int(max(1, round(npart / fsample)))
                mass_weight = fsample

            nsample = min(nsample, npart)
            index_all = np.arange(npart, dtype=int)
            index_sample = random.sample(index_all.tolist(), nsample)

            self.mass = mass[index_sample] * 1e10 * mass_weight
            self.pos = pos[index_sample, :] * 1e-3
            self.ptype = ptype[index_sample]
            self.pid = pid[index_sample]
            if readpot:
                self.pot = pot[index_sample]
        else:
            self.mass=mass*1e10#[index_sample]*1e10 # particle masses in Msun/h
            self.pos=pos*1e-3#[index_sample,:]*1e-3 # particle comoving coordinates in Mpc/h
            self.ptype=ptype
            if readpot:
                self.pot=pot
            self.pid=pid
        # read info from the header of the snapshot file and store into cluster instance
        if hdf5:
            self.zl=sim_file['Header'].attrs['Redshift']
            self.omega=sim_file['Header'].attrs['Omega0']
            self.lambd=sim_file['Header'].attrs['OmegaLambda']
            self.h=sim_file['Header'].attrs['HubbleParam']
        else:

            #header=G.snapshot_header(self.snapshot)
            header = g3read.GadgetFile(self.snapshot).header
            self.zl=header.redshift
            self.omega=header.Omega0
            self.lambd=header.OmegaLambda
            self.h=header.HubbleParam
        self.co = FlatLambdaCDM(H0=self.h*100.0, Om0=self.omega)

        # now converting comoving into physical units
        self.pos=self.pos/(1.0+self.zl)/self.h # particle coordinates are converted into physical Mpc
        self.mass=self.mass/self.h # particle masses: from Msun/h to Msun
        self.sn=sn
        self.cln=cln
        # convert center positions in physical coordinates
        self.xc=xc/(1.0+self.zl)/self.h # center coordinates in physical Mpc
        self.yc=yc/(1.0+self.zl)/self.h
        self.zc=zc/(1.0+self.zl)/self.h

        if alignIT:
            self.pos = np.matmul(self.pos,IT)
            self.xc, self.yc, self.zc = np.matmul(np.array([self.xc,self.yc,self.zc]),IT)

    def massMap(self, pX=0.0, pY=0.0, npix=512,
        xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0, zmin=-3.0, zmax=3.0, sigma=3):
        """
        Generate a 2D mass map of the cluster by projecting particles onto a grid and smoothing with a Gaussian kernel.

        Args:
            pX (float): Rotation around X axis before projecting along Z.
            pY (float): Rotation around Y axis before projecting along Z.
            npix (int): Number of pixels along each axis (square map).
            xmin, xmax, ymin, ymax, zmin, zmax (float): Volume boundaries [Mpc].
            sigma (float): Sigma of the Gaussian smoothing kernel.

        Returns:
            np.ndarray: 2D map of cluster mass distribution in the selected region [Msun].
        """
        
        # recenter particles and apply rotations and then project along the 'z' axis
        self.recenter(self.xc,self.yc,self.zc)
        self.rotateX(pX)
        self.rotateY(pY)
        self.angx=pX
        self.angy=pY
        x_tmp=self.pos[:,0]
        y_tmp=self.pos[:,1]
        # select the particles in the specified region
        isel=((self.pos[:,2] >= zmin) & (self.pos[:,2] <= zmax)
             & (self.pos[:,0] >= xmin) & (self.pos[:,0] <= xmax)
             & (self.pos[:,1] >= ymin) & (self.pos[:,1] <= ymax))
        x=x_tmp[isel]
        y=y_tmp[isel]
        mass=self.mass[isel]
        if self.verbose:
            print ('selected #particles:',len(mass))
            print ('total selected mass:',mass.sum())
        #'''
        
        # produce the mass map and apply gaussian filter
        xi = np.linspace(xmin, xmax, npix+1)
        yi = np.linspace(ymin, ymax, npix+1)
        #massmap, xedges, yedges = np.histogram2d(x, y, bins=(xi, yi),weights=mass)
        massmap, xedges, yedges = np.histogram2d(y, x, bins=(xi, yi), weights=mass)
        summa=np.sum(massmap)
        summam=np.sum(mass)

        #print ('map normalization:',summa,summam)
        if summa > 0.0:
            massmap=massmap/summa*summam
        else:
            massmap = np.zeros_like(massmap)
        img = ndimage.gaussian_filter(massmap, sigma=(sigma, sigma), order=0)
        
        self.reset()
        return(img)

    def massMapSPH(self, pX=0.0, pY=0.0, npix=512,
        xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0, zmin=-3.0, zmax=3.0, nb=4,
        min_hsml=1e-5, max_hsml=0.1):
        """
        Generate a 2D mass map using SPH kernel projection.

        If legacy ``sphviewer`` is installed, this method uses ``QuickView``.
        If ``py-sphviewer2`` is installed instead, it uses the compatible
        ``sphviewer2`` API. Otherwise it automatically falls back to
        ``massMapSPH_internal``.

        Args:
            pX (float): Rotation around X axis before projecting along Z.
            pY (float): Rotation around Y axis before projecting along Z.
            npix (int): Number of pixels along each axis (square map).
            xmin, xmax, ymin, ymax, zmin, zmax (float): Volume boundaries [Mpc].
            nb (int): Number of neighboring particles for SPH kernel.
            min_hsml, max_hsml (float): Min/max smoothing length.

        Returns:
            np.ndarray: 2D map of cluster mass distribution in the selected region [Msun].
        """
        if not _SPHVIEWER_AVAILABLE:
            if self.verbose:
                print('massMapSPH: sphviewer not available, using massMapSPH_internal.')
            return self.massMapSPH_internal(
                pX=pX, pY=pY, npix=npix,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                zmin=zmin, zmax=zmax, nb=nb,
                min_hsml=min_hsml, max_hsml=max_hsml
            )

        # Recenter particles
        self.recenter(self.xc,self.yc,self.zc)

        # apply rotations and then project along the 'z' axis
        self.rotateX(pX)
        self.rotateY(pY)
        self.angx=pX
        self.angy=pY
        # select the particles in the specified region
        isel=((self.pos[:,2] >= zmin) & (self.pos[:,2] <= zmax)
             & (self.pos[:,0] >= xmin) & (self.pos[:,0] <= xmax)
             & (self.pos[:,1] >= ymin) & (self.pos[:,1] <= ymax))
       
        pos=self.pos[isel,:]
        mass=self.mass[isel]
        if mass.size == 0:
            self.reset()
            return np.zeros((npix, npix), dtype=np.float64)
        #print ('selected #particles:',len(mass))
        if _SPHVIEWER_BACKEND == "sphviewer":
            qv = QuickView(pos, mass, r='infinity', nb=nb, plot=False, xsize=npix, ysize=npix, logscale=False,
                           min_hsml=min_hsml, max_hsml=max_hsml)
            #qv=QuickView(pos.T, mass, r='infinity', nb=nb, plot=False, xsize=npix, ysize=npix, logscale=False)
            img=qv.get_image()
        elif _SPHVIEWER_BACKEND == "sphviewer2":
            img = _mass_map_sphviewer2(
                pos, mass, npix=npix,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                zmin=zmin, zmax=zmax, nb=nb,
                min_hsml=min_hsml, max_hsml=max_hsml
            )
        else:
            img = self.massMapSPH_internal(
                pX=0.0, pY=0.0, npix=npix,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                zmin=zmin, zmax=zmax, nb=nb,
                min_hsml=min_hsml, max_hsml=max_hsml
            )
        if img.sum() > 0.0:
            img=img/img.sum()*mass.sum() # normalization, to ensure that the mass is conserved
        self.reset()
        return(img)

    def massMapSPH_internal(self, pX=0.0, pY=0.0, npix=512,
        xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0, zmin=-3.0, zmax=3.0, nb=4,
        min_hsml=1e-5, max_hsml=0.1, use_numba=True, local_mass_conservation=False,
        kernel="cubic_spline", kernel_gamma=None):
        """
        Generate a 2D mass map using an internal SPH kernel projection.

        This method is an in-package alternative to ``massMapSPH`` (py-sphviewer)
        and is designed for large particle sets by combining:
        - kNN smoothing-length estimation via ``cKDTree``
        - compact-support cubic-spline or Wendland-C2 SPH kernel
        - optional numba-compiled deposition loop

        Parameters
        ----------
        pX, pY : float
            Rotations around X and Y axis before projection.
        npix : int
            Number of pixels along each axis.
        xmin, xmax, ymin, ymax, zmin, zmax : float
            Selection box boundaries in Mpc.
        nb : int
            Number of neighbors for adaptive smoothing length.
        min_hsml, max_hsml : float
            Lower/upper bounds for smoothing lengths in Mpc.
        use_numba : bool
            If True and numba is available, use the compiled backend.
        local_mass_conservation : bool
            If True, enforce per-particle discrete normalization (slower).
            If False, use one-pass deposition with global renormalization (faster,
            and closer to the old ``massMapSPH`` behavior).
        kernel : {"cubic_spline", "wendland_c2"}
            Compact-support 2D SPH kernel used for the internal deposition.
            ``"cubic_spline"`` preserves the historical behavior. ``"wendland_c2"``
            uses a non-negative Wendland-C2 profile.
        kernel_gamma : float or None
            Compact-support radius in units of smoothing length for the
            Wendland-C2 kernel. If None, use the SWIFT/SWIFTSIMIO Wendland-C2
            value ``1.936492``. The cubic-spline kernel always uses ``2h``.

        Returns
        -------
        ndarray
            SPH-smoothed projected mass map in Msun.
        """
        self.recenter(self.xc, self.yc, self.zc)
        self.rotateX(pX)
        self.rotateY(pY)
        self.angx = pX
        self.angy = pY

        try:
            isel = ((self.pos[:, 2] >= zmin) & (self.pos[:, 2] <= zmax)
                 & (self.pos[:, 0] >= xmin) & (self.pos[:, 0] <= xmax)
                 & (self.pos[:, 1] >= ymin) & (self.pos[:, 1] <= ymax))

            pos = self.pos[isel, :]
            mass = self.mass[isel]
            if self.verbose:
                print('selected #particles:', len(mass))
                print('total selected mass:', mass.sum())
                if use_numba and not _NUMBA_AVAILABLE:
                    print('massMapSPH_internal: numba not available, using slower Python backend.')

            img = _mass_map_sph_native(
                pos=pos, mass=mass, npix=npix,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                nb=nb, min_hsml=min_hsml, max_hsml=max_hsml,
                use_numba=use_numba,
                local_mass_conservation=local_mass_conservation,
                kernel=kernel,
                kernel_gamma=kernel_gamma
            )
        finally:
            self.reset()
        return img

    def massMapSPH_tmp(self, pX=0.0, pY=0.0, npix=512,
        xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0, zmin=-3.0, zmax=3.0, nb=4,
        min_hsml=1e-5, max_hsml=0.1, use_numba=True, local_mass_conservation=False,
        kernel="cubic_spline", kernel_gamma=None):
        """
        Deprecated alias for ``massMapSPH_internal``.
        """
        warnings.warn(
            "massMapSPH_tmp is deprecated; use massMapSPH_internal instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.massMapSPH_internal(
            pX=pX, pY=pY, npix=npix,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            zmin=zmin, zmax=zmax, nb=nb,
            min_hsml=min_hsml, max_hsml=max_hsml,
            use_numba=use_numba,
            local_mass_conservation=local_mass_conservation,
            kernel=kernel,
            kernel_gamma=kernel_gamma,
        )

    def selSave2ascii(self, pX=0.0, pY=0.0,
                      xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0, zmin=-3.0, zmax=3.0, outputn='sim.txt'):
        """
        Select particles in a volume and save their positions, masses, and types to an ASCII file.

        Args:
            pX (float): Rotation around X axis before projecting along Z.
            pY (float): Rotation around Y axis before projecting along Z.
            xmin, xmax, ymin, ymax, zmin, zmax (float): Volume boundaries [Mpc].
            outputn (str): Output filename for particle data.

        Returns:
            None
        """

        self.recenter(self.xc,self.yc,self.zc)

        self.rotateX(pX)
        self.rotateY(pY)
        self.angx=pX
        self.angy=pY
        # select the particles in the specified region
        isel=((self.pos[:,2] >= zmin) & (self.pos[:,2] <= zmax)
             & (self.pos[:,0] >= xmin) & (self.pos[:,0] <= xmax)
             & (self.pos[:,1] >= ymin) & (self.pos[:,1] <= ymax))
       
        pos=self.pos[isel,:]
        mass=self.mass[isel]
        ptype=self.ptype[isel]
        if self.verbose:
            print ('selected #particles:',len(mass))
        arr=zip(pos[:,0],pos[:,1],pos[:,2],mass,ptype)
        np.savetxt(outputn,arr,fmt='%12f,%12f,%12f,%12e,%i')

    
    def rotateX(self, angle):
        """
        Rotate particles by an angle around the X axis.

        Args:
            angle (float): Rotation angle in degrees.

        Returns:
            None
        """
        y=self.pos[:,1]
        z=self.pos[:,2]
        rad = angle * np.pi / 180
        cosa = np.cos(rad)
        sina = -np.sin(rad)
        yy= y * cosa - z * sina
        zz =y * sina + z * cosa
        self.pos[:,1]=yy
        self.pos[:,2]=zz

    
    def rotateY(self, angle):
        """
        Rotate particles by an angle around the Y axis.

        Args:
            angle (float): Rotation angle in degrees.

        Returns:
            None
        """
        x=self.pos[:,0]
        z=self.pos[:,2]
        rad = angle * np.pi / 180
        cosa = np.cos(rad)
        sina = -np.sin(rad)
        zz = z * cosa - x * sina
        xx = z * sina + x * cosa
        self.pos[:,0]=xx
        self.pos[:,2]=zz
    
    def recenter(self, xc, yc, zc):
        """
        Re-center the particles at the specified position.

        Args:
            xc (float): New x coordinate of the center [Mpc].
            yc (float): New y coordinate of the center [Mpc].
            zc (float): New z coordinate of the center [Mpc].

        Returns:
            None
        """
        self.pos[:,0]=self.pos[:,0]-xc
        self.pos[:,1]=self.pos[:,1]-yc
        self.pos[:,2]=self.pos[:,2]-zc

    
    def reset(self):
        """
        Undo previous rotations and recenter the cluster.

        Returns:
            None
        """
        self.rotateY(-self.angy)
        self.rotateX(-self.angx)
        self.recenter(-self.xc,-self.yc,-self.zc)

    def center_around(self, x, y, z, radius):
        """
        Find the particle with the minimum potential within a spherical region.

        Args:
            x (float): X coordinate of the sphere center.
            y (float): Y coordinate of the sphere center.
            z (float): Z coordinate of the sphere center.
            radius (float): Radius of the sphere.

        Returns:
            int: Index of the particle with minimum potential (if applicable).
        """
        d2=(self.pos[:,0]-x)**2 + (self.pos[:,1]-y)**2 + (self.pos[:,2]-z)**2
        isel = d2<radius**2
        pot_tmp=self.pot[isel]
        pos_tmp=self.pos[isel,:]
        ii=np.argmin(pot_tmp)
        return (pos_tmp[ii,0],pos_tmp[ii,1],pos_tmp[ii,2])
