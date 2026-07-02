import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from pyLensLib.spiral import spiral
from pyLensLib.sersic_numba import sersic
from pyLensLib.perlin import generate_perlin_noise

# Constants
npix = 1000
sizex = [-5., 5.]
sizey = [-5., 5.]
kwargs_disc = {
    'n': 1.0,
    'q': 1.0,
    'ys1': 0.0,
    'ys2': 0.0,
    'pa': np.deg2rad(45.0),
    're': 0.3,
    'flux': 1.0,
    'zs': 1.0
}
kwargs_bulge = {
    'n': 8.0,
    'q': 1.0,
    'ys1': 0.0,
    'ys2': 0.0,
    'pa': np.deg2rad(0.0),
    're': 0.1,
    'flux': 2.0,
    'zs': 1.0
}
A, Na, phid, alpha = 5.0, 2, 100, 10.1

# Precompute Perlin noise
noise_map = generate_perlin_noise(npix, npix, scale=10, octaves=4, persistence=1.5, lacunarity=2.0, seed=42)
noise_bulge = generate_perlin_noise(npix, npix, scale=10, octaves=3, persistence=1.5, lacunarity=2.0, seed=10)

# Generate Sersic profiles
se = sersic(Npix=npix, gl=None, save_unlensed=True, rmaxf=10.0, sizex=sizex, sizey=sizey, **kwargs_disc)
se_bulge = sersic(Npix=npix, gl=None, save_unlensed=True, rmaxf=10.0, sizex=sizex, sizey=sizey, **kwargs_bulge)
se_bulge_image = se_bulge.image * noise_bulge
se_bulge_image /= np.sum(se_bulge_image) * kwargs_bulge['flux']

# Animation setup
phi_list = np.linspace(0, np.pi, 200)
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((npix, npix)), animated=True, cmap='gray_r', norm=LogNorm())

def update(frame):
    phi = phi_list[frame]
    sp = spiral(se, A=A, Na=Na, Phid=phid, alpha=alpha, phi=phi, z0=0.05)
    sp_image = sp.brightness() * noise_map
    sp_image /= np.sum(sp_image) * kwargs_disc['flux']
    snapshot = sp_image + se_bulge_image
    im.set_array(snapshot)
    return [im]

ani = FuncAnimation(fig, update, frames=len(phi_list), blit=True, interval=50)
ani.save('galaxy_rotation.mp4', writer='ffmpeg')
plt.close(fig)