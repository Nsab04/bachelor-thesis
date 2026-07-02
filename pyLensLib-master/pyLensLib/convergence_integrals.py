import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


def _centered_coordinate_grid(num_pix, delta_pix):
    """
    Return centered 2D coordinate grids with lenstronomy-compatible orientation.
    """
    if num_pix % 2 == 0:
        raise ValueError("num_pix must be odd so the kernel has a defined center.")
    axis = (np.arange(num_pix, dtype=float) - (num_pix - 1) / 2.0) * delta_pix
    return np.meshgrid(axis, axis)


def _kernel_size_for_map(kappa):
    num_pix = int(np.shape(kappa)[0]) * 2
    if num_pix % 2 == 0:
        num_pix += 1
    return num_pix


def _validate_square_grid(kappa):
    kappa = np.asarray(kappa, dtype=float)
    if kappa.ndim != 2 or kappa.shape[0] != kappa.shape[1]:
        raise ValueError("kappa must be a square 2D array.")
    return kappa


def re_size(image, factor):
    """
    Downsample a 2D image by block averaging.
    """
    factor = int(factor)
    image = _validate_square_grid(image)
    if factor <= 1:
        return image.copy()
    if image.shape[0] % factor != 0:
        raise ValueError("image size must be divisible by factor.")
    new_size = image.shape[0] // factor
    return image.reshape(new_size, factor, new_size, factor).mean(axis=(1, 3))


def deflection_kernel(num_pix, delta_pix):
    """
    Numerical Green's-function kernel for deflection from convergence.

    Parameters
    ----------
    num_pix : int
        Odd number of pixels per kernel axis.
    delta_pix : float
        Pixel spacing in angular units.

    Returns
    -------
    tuple of ndarray
        Kernels for the x and y deflection components.
    """
    x_shift, y_shift = _centered_coordinate_grid(int(num_pix), float(delta_pix))
    r2 = x_shift * x_shift + y_shift * y_shift
    r2[r2 < (delta_pix / 2.0) ** 2] = (delta_pix / 2.0) ** 2
    return x_shift / r2, y_shift / r2


def potential_kernel(num_pix, delta_pix):
    """
    Numerical Green's-function kernel for lensing potential from convergence.
    """
    x_shift, y_shift = _centered_coordinate_grid(int(num_pix), float(delta_pix))
    r2 = x_shift * x_shift + y_shift * y_shift
    r2_max = np.max(r2)
    r2[r2 < (delta_pix / 2.0) ** 2] = (delta_pix / 2.0) ** 2
    return 0.5 * np.log(r2 / r2_max)


def deflection_from_kappa_grid(kappa, grid_spacing):
    """
    Compute deflection angles from a convergence grid by FFT convolution.

    ``kappa`` is dimensionless and ``grid_spacing`` sets the angular unit of the
    returned deflection maps.
    """
    kappa = _validate_square_grid(kappa)
    num_pix = _kernel_size_for_map(kappa)
    kernel_x, kernel_y = deflection_kernel(num_pix, grid_spacing)
    scale = grid_spacing ** 2 / np.pi
    f_x = fftconvolve(kappa, kernel_x, mode="same") * scale
    f_y = fftconvolve(kappa, kernel_y, mode="same") * scale
    return f_x, f_y


def kappa_from_surface_density(surface_density, sigma_crit):
    """
    Convert a surface-density grid to convergence.

    ``surface_density`` and ``sigma_crit`` must be in consistent mass-per-area
    units, for example Msun/Mpc^2.
    """
    return np.asarray(surface_density, dtype=float) / float(sigma_crit)


def deflection_from_surface_density_grid(
    surface_density,
    sigma_crit,
    grid_spacing,
    adaptive=False,
    low_res_factor=4,
    high_res_kernel_size=21,
    return_high_res=False,
):
    """
    Compute deflection angles directly from a surface-density map.
    """
    kappa = kappa_from_surface_density(surface_density, sigma_crit)
    if adaptive:
        return deflection_from_kappa_grid_adaptive(
            kappa,
            grid_spacing,
            low_res_factor=low_res_factor,
            high_res_kernel_size=high_res_kernel_size,
            return_high_res=return_high_res,
        )
    return deflection_from_kappa_grid(kappa, grid_spacing)


def potential_from_kappa_grid(kappa, grid_spacing):
    """
    Compute the lensing potential from a convergence grid by FFT convolution.
    """
    kappa = _validate_square_grid(kappa)
    num_pix = _kernel_size_for_map(kappa)
    kernel = potential_kernel(num_pix, grid_spacing)
    return fftconvolve(kappa, kernel, mode="same") * grid_spacing ** 2 / np.pi


def _central_square_kernel(kernel, size):
    size = int(size)
    if size < 1:
        raise ValueError("kernel size must be positive.")
    if size % 2 == 0:
        size += 1
    if size > kernel.shape[0]:
        size = kernel.shape[0]
    if size % 2 == 0:
        size -= 1
    out = np.zeros_like(kernel)
    center = kernel.shape[0] // 2
    radius = size // 2
    out[center - radius:center + radius + 1,
        center - radius:center + radius + 1] = (
            kernel[center - radius:center + radius + 1,
                   center - radius:center + radius + 1]
        )
    return out


def _kernel_with_central_hole(kernel, hole_size):
    hole_size = int(hole_size)
    if hole_size % 2 == 0:
        hole_size += 1
    out = kernel.copy()
    center = out.shape[0] // 2
    radius = min(hole_size // 2, center)
    out[center - radius:center + radius + 1,
        center - radius:center + radius + 1] = 0.0
    return out


def deflection_from_kappa_grid_adaptive(
    kappa_high_res,
    grid_spacing,
    low_res_factor,
    high_res_kernel_size,
    return_high_res=False,
):
    """
    Approximate split-resolution FFT deflection calculation.

    The near-field kernel is evaluated on the high-resolution map and the
    far-field kernel is evaluated on a block-averaged lower-resolution map.
    By default the result is returned on the lower-resolution grid, matching
    lenstronomy's adaptive convergence-integral convention.
    """
    kappa_high_res = _validate_square_grid(kappa_high_res)
    low_res_factor = int(low_res_factor)
    if low_res_factor <= 1:
        return deflection_from_kappa_grid(kappa_high_res, grid_spacing)

    kappa_low_res = re_size(kappa_high_res, low_res_factor)
    grid_spacing_low_res = grid_spacing * low_res_factor

    num_pix_high = _kernel_size_for_map(kappa_high_res)
    high_size = int(high_res_kernel_size) * low_res_factor
    kernel_high_x, kernel_high_y = deflection_kernel(num_pix_high, grid_spacing)
    kernel_high_x = _central_square_kernel(kernel_high_x, high_size)
    kernel_high_y = _central_square_kernel(kernel_high_y, high_size)

    scale_high = grid_spacing ** 2 / np.pi
    f_x_high = fftconvolve(kappa_high_res, kernel_high_x, mode="same") * scale_high
    f_y_high = fftconvolve(kappa_high_res, kernel_high_y, mode="same") * scale_high

    num_pix_low = _kernel_size_for_map(kappa_low_res)
    hole_size = int(high_res_kernel_size)
    kernel_low_x, kernel_low_y = deflection_kernel(num_pix_low, grid_spacing_low_res)
    kernel_low_x = _kernel_with_central_hole(kernel_low_x, hole_size)
    kernel_low_y = _kernel_with_central_hole(kernel_low_y, hole_size)

    scale_low = grid_spacing_low_res ** 2 / np.pi
    f_x_low = fftconvolve(kappa_low_res, kernel_low_x, mode="same") * scale_low
    f_y_low = fftconvolve(kappa_low_res, kernel_low_y, mode="same") * scale_low

    if return_high_res:
        zoom = float(low_res_factor)
        f_x_low = ndimage.zoom(f_x_low, zoom=zoom, order=1)[: kappa_high_res.shape[0], : kappa_high_res.shape[1]]
        f_y_low = ndimage.zoom(f_y_low, zoom=zoom, order=1)[: kappa_high_res.shape[0], : kappa_high_res.shape[1]]
        return f_x_high + f_x_low, f_y_high + f_y_low

    f_x_high = re_size(f_x_high, low_res_factor)
    f_y_high = re_size(f_y_high, low_res_factor)
    return f_x_high + f_x_low, f_y_high + f_y_low


def potential_from_kappa_grid_adaptive(
    kappa_high_res,
    grid_spacing,
    low_res_factor,
    high_res_kernel_size,
    return_high_res=False,
):
    """
    Approximate split-resolution FFT potential calculation.
    """
    kappa_high_res = _validate_square_grid(kappa_high_res)
    low_res_factor = int(low_res_factor)
    if low_res_factor <= 1:
        return potential_from_kappa_grid(kappa_high_res, grid_spacing)

    kappa_low_res = re_size(kappa_high_res, low_res_factor)
    grid_spacing_low_res = grid_spacing * low_res_factor

    kernel_high = potential_kernel(_kernel_size_for_map(kappa_high_res), grid_spacing)
    kernel_high = _central_square_kernel(
        kernel_high, int(high_res_kernel_size) * low_res_factor
    )
    f_high = fftconvolve(kappa_high_res, kernel_high, mode="same") * grid_spacing ** 2 / np.pi

    kernel_low = potential_kernel(_kernel_size_for_map(kappa_low_res), grid_spacing_low_res)
    kernel_low = _kernel_with_central_hole(kernel_low, int(high_res_kernel_size))
    f_low = fftconvolve(kappa_low_res, kernel_low, mode="same") * grid_spacing_low_res ** 2 / np.pi

    if return_high_res:
        f_low = ndimage.zoom(f_low, zoom=float(low_res_factor), order=1)[
            : kappa_high_res.shape[0], : kappa_high_res.shape[1]
        ]
        return f_high + f_low

    return re_size(f_high, low_res_factor) + f_low
