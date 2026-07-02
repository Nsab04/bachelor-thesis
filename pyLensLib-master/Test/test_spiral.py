from pyLensLib.spiral import spiral
from pyLensLib.sersic_numba import sersic
from pyLensLib.perlin import generate_perlin_noise
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

npix=1000
sizex = [-5., 5.]
sizey = [-5., 5.]
fsize = sizex[1] - sizex[0]



# Generating the disc involves two steps: first generate an Exponentialdisc (Sersic with n=1)
kwargs_disc = {
    'n': 1.0,
    'q': 1.0,
    'ys1': 0.0,  # ys1+dy1,
    'ys2': 0.0,  # ys2+dy2,
    'pa': np.deg2rad(45.0),
    're': 0.3,
    'flux': 1.0,
    'zs': 1.0
}

se = sersic(Npix=npix, gl=None, save_unlensed=True,
	rmaxf=10.0, sizex=sizex, sizey=sizey, **kwargs_disc)

A = 5.0
Na = 2
phid = 100
alpha = 10.1
phi = np.deg2rad(0.0) # inclination angle
sp = spiral(se, A=A, Na=Na, Phid=phid, alpha=alpha, phi=phi)
sp_image = sp.brightness()
print ('Created spiral galaxy with inclination angle (deg):', np.rad2deg(phi))

# perlin noise for the disc
noise_map = generate_perlin_noise(npix, npix, scale=10, octaves=4, persistence=1.5, lacunarity=2.0, seed=42)
sp_image*=noise_map
sp_image=sp_image/np.sum(sp_image)*kwargs_disc['flux']
print ('Added perlin noise to the disc')

# bulge
kwargs_bulge = {
    'n': 2.0,
    'q': 1.0,
    'ys1': 0.0,  # ys1+dy1,
    'ys2': 0.0,  # ys2+dy2,
    'pa': np.deg2rad(0.0),
    're': 0.01,
    'flux': 0.1,
    'zs': 1.0
}
se_bulge = sersic(Npix=npix, gl=None, save_unlensed=True,
	rmaxf=10.0, sizex=sizex, sizey=sizey, **kwargs_bulge)
noise_bulge = generate_perlin_noise(npix, npix, scale=10, octaves=3, persistence=1.5, lacunarity=2.0, seed=10)
se_bulge_image = se_bulge.image*noise_bulge

print ('Added perlin noise to the bulge')

se_bulge_image = se_bulge_image/np.sum(se_bulge_image) * kwargs_bulge['flux']
fig,ax = plt.subplots(1,3,figsize=(30,10))
ax[0].imshow(se.image)
ax[1].imshow(sp_image+se_bulge.image,cmap='gray_r',norm=LogNorm())

phi = np.deg2rad(90.0) # inclination angle
sp = spiral(se, A=A, Na=Na, Phid=phid, alpha=alpha, phi=phi)
sp_image = sp.brightness()

# perlin noise for the disc
noise_map = generate_perlin_noise(npix, npix, scale=10, octaves=4, persistence=1.5, lacunarity=2.0, seed=42)
sp_image*=noise_map
sp_image=sp_image/np.sum(sp_image)*kwargs_disc['flux']
ax[2].imshow(sp_image+se_bulge.image,cmap='gray_r',norm=LogNorm())

fig.savefig('spiral_galaxy_examples.png')
plt.close(fig)


# now create several snapshots of the galaxy with different inclinations and create an animation
phi_list = np.linspace(0, 2.0*np.pi, 300)

print ('Creating animation of galaxy rotation...')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
frames = []
fig, ax = plt.subplots()

for phi in phi_list:
    print ('Inclination angle (deg):', np.rad2deg(phi))
    sp = spiral(se, A=A, Na=Na, Phid=phid, alpha=alpha, phi=phi, z0=0.05)
    sp_image = sp.brightness()
    sp_image *= noise_map
    sp_image = sp_image / np.sum(sp_image) * kwargs_disc['flux']
    snapshot = sp_image + se_bulge.image

    # Append the image to the frames list
    frame = ax.imshow(snapshot, animated=True, cmap='gray_r',norm=LogNorm())
    frames.append([frame])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)

# Save the animation
ani.save('galaxy_rotation.mp4', writer='ffmpeg')

print ('Animation saved as galaxy_rotation.mp4')
# Or display the animation
#plt.show()


