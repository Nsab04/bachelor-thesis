# generate a resolution test for mass mapping
# create a 3D NFW halo containing a subhalo. Sample it with particles.
# create 2D maps analytically and by projecting the data cube

import numpy as np
from pyLensLib.piemd import piemd
from astropy.cosmology import FlatLambdaCDM
from sphviewer.tools import QuickView

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

# create a datacube

n=512
side = 25.0
p = side/(n-1) # in arcsec

# 3D coordinates and grids
X, Y, Z = np.ogrid[0:n, 0:n, 0:n]
X = p*(X - n/2.0 + 0.5)
Y = p*(Y - n/2.0 + 0.5)
Z = p*(Z - n/2.0 + 0.5)
d = np.sqrt(X*X+Y*Y+Z*Z) + 1e-3

# 2D coordinates and grids
x, y = np.ogrid[0:n, 0:n]
x = p*(x - n/2.0 + 0.5)
y = p*(y - n/2.0 + 0.5)
r = np.sqrt(x*x+y*y) + 1e-3

print (r.shape)

# PIEMD lens
kwargs = {'zl': 0.5,
          'zs': 3.0,
          'sigma0': 120.0,
          'q': 1.0,
          'pa': 0.0,
          'theta_c': 1e-3,
          'theta_t': 1.0,
          'x1': 0.0,
          'x2': 0.0}

dpie = piemd(co, **kwargs)

# compute 3D density
den = dpie.density(d)
cube = den.reshape(n,n,n)

pv = np.deg2rad(p/3600.0)*dpie.dl.value

# compute 2D density
sd = dpie.surf_density(r)
surf = sd.reshape(n,n)



import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,4,figsize=(32,10))

for i in range(3):
    proj_cube = cube.sum(axis=i)*pv
    ax[i].imshow(np.log10(proj_cube),origin='lower',vmax=14)
    ax[i].contour(np.log10(proj_cube),levels=[11],colors='orange')

ax[3].imshow(np.log10(surf),origin='lower',vmax=14)
ax[3].contour(np.log10(surf),levels=[11])
ax[2].contour(np.log10(surf),levels=[11],colors='yellow',linestyles='--')

print (('Total 3D mass: %10.4e ') % (np.sum(cube)*pv**3))
print (('Total 3D mass (proj): %10.4e ') % (np.sum(proj_cube)*pv**2))
proj_cube[r<p] = 0.0
print (('Total 3D mass (proj., core exc.): %10.4e ') % (np.sum(proj_cube)*pv**2))
print (('Total 2D mass: %10.4e ') % (np.sum(surf)*pv**2))
surf[r<p] = 0.0
print (('Total 2D mass (core exc.): %10.4e ') % (np.sum(surf)*pv**2))

import pyLensLib.samplers as sa

mp=3e6
npart = int(np.sum(cube)*pv**3/mp)+1
print ('Number of particles:',npart)

xp,yp,zp = sa.sample3Dcube(cube,n=npart)
ax[0].plot(xp,yp,',',color='red')
ax[1].plot(xp,zp,',',color='red')
ax[2].plot(yp,zp,',',color='red')

#massbkg=1e13
#npart_bkg = int(massbkg/mp)
#xb = np.random.rand(npart_bkg)*n
#yb = np.random.rand(npart_bkg)*n
#zb = np.random.rand(npart_bkg)*n

#ax[0].plot(xb,yb,',',color='orange')
#ax[1].plot(xb,zb,',',color='orange')
#ax[2].plot(yb,zb,',',color='orange')

plt.show()

pos = np.stack((xp*p-side/2., yp*p-side/2., zp*p-side/2.), axis=1)
#pos_bkg = np.stack((xb*p-side/2., yb*p-side/2., zb*p-side/2.), axis=1)

#pos_all = np.concatenate((pos,pos_bkg),axis=0)
#npart_all = npart + npart_bkg

m = np.ones(npart)*mp
nb=30
min_hsml=0.001
max_hsml=0.1
zl=0.5
nray=2048

qv = QuickView(pos, m, r='infinity', nb=nb, plot=False, xsize=nray, ysize=nray, logscale=False,
                       min_hsml=min_hsml, max_hsml=max_hsml)

img=qv.get_image()
img=img/img.sum()*m.sum()

from pyLensLib.raytracer import raytracer
from  pyLensLib.deflector import deflector

fov_ray_arcsec = side


kwargs = {'zl': zl, 'zs': 3.0, 'fov': fov_ray_arcsec}
rayt=raytracer(co,img,Nray=nray,FOVray=fov_ray_arcsec,fromfile=False,**kwargs)
kwargs_def={'zl': zl, 'zs': kwargs['zs']}
df_simu=deflector(co,angx=rayt.a1,angy=rayt.a2,**kwargs_def)
theta=np.linspace(-fov_ray_arcsec/2.0,fov_ray_arcsec/2.0,nray)
df_simu.setGrid(theta=theta,compute_potential=False)
df_simu.ka[df_simu.ka<0]=0.0

print (m.sum(),img.max(),img.min())

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(img,origin='lower',vmax=np.max(img)*0.5,extent=[-fov_ray_arcsec/2.,fov_ray_arcsec/2.,-fov_ray_arcsec/2.,fov_ray_arcsec/2.])

tl=df_simu.tancl()
print (tl[0].getThetaE()*df_simu.pixel_scale)
rl=df_simu.radcl()
ctl=df_simu.getCaustics(tl)
crl=df_simu.getCaustics(rl)
for c in tl:
    xc,yc=df_simu.getCritPoints(c,pixel_units=False)
    if (c.principale):
        ax.plot(xc,yc,'-',color='white')
    else:
        ax.plot(xc,yc,'-',color='yellow')
for c in rl:
    xc,yc=df_simu.getCritPoints(c,pixel_units=False)
    ax.plot(xc,yc,'-',color='white')

dpie.setGrid(theta=theta)
tl=dpie.tancl()
print (tl[0].getThetaE()*df_simu.pixel_scale)
rl=dpie.radcl()
ctl=dpie.getCaustics(tl)
crl=dpie.getCaustics(rl)
for c in tl:
    xc,yc=dpie.getCritPoints(c,pixel_units=False)
    if (c.principale):
        ax.plot(xc,yc,'-',color='red')
    else:
        ax.plot(xc,yc,'-',color='orange')
for c in rl:
    xc,yc=dpie.getCritPoints(c,pixel_units=False)
    ax.plot(xc,yc,'-',color='red')

plt.show()



show3d = False
if show3d:

    from vispy import app, scene, visuals
    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
    canvas = scene.SceneCanvas(keys='interactive', show=True,bgcolor='white')
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 10
    view.camera.set_range(x=(0,n-1),y=(0,n-1),z=(0,n-1))

    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.set_data(pos, face_color='red', symbol='o', size=2,
                edge_width=0.05, edge_color='blue')

    canvas.show()
    app.run()