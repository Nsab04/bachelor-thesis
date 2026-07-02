from pyLensLib import lenstool as lst
import numpy as np
import matplotlib.pyplot as plt
from pyLensLib.sersic_numba import sersic
from pyLensLib.pointsrc import pointsrc
from pyLensLib.observation import observation

zcl = lst.getLensRedshift('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo.par')
#df = lst.create_deflector('/Users/maxmen3/stiva/pietro_models/M0416_B22_large.par',
#                          filex='/Users/maxmen3/stiva/pietro_models/M0416_B22_large_angx.fits',
#                          filey='/Users/maxmen3/stiva/pietro_models/M0416_B22_large_angy.fits',zl=0.396,zsnorm=1.0,zs=6.145)

df = lst.create_deflector('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo.par',
                          filex='/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo_angx.fits',
                          filey='/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo_angy.fits',zl=zcl,zsnorm=1.0,zs=3.0)

# get the WCS from one of the deflection angle maps
from astropy.io import fits
from astropy.wcs import WCS
#hdul = fits.open('/Users/maxmen3/stiva/pietro_models/M0416_B22_large_angx.fits')
hdul = fits.open('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo_angx.fits')
hdr = hdul[0].header
wcs = WCS(hdr)

#fov = lst.getFoV('/Users/maxmen3/stiva/pietro_models/M0416_B22_large.par')
fov = lst.getFoV('/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/elgordo.par')
fov_side = fov[1]-fov[0]
print (fov,fov_side)
extent = (-fov_side/2.0, fov_side/2.0, -fov_side/2.0, fov_side/2.0)

theta1 = -35.56908
theta2 = 18.8532

a1, a2 = df.getAngle(theta1,theta2)
kappa = df.ka
gamma1, gamma2 = df.g1, df.g2
detA = (1.0-kappa)**2 - gamma1**2 - gamma2**2

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(1.0/abs(detA),vmax=50.0,vmin=1.0,origin='lower',extent=extent)
ax.contour(detA, levels=[0.0], colors='red', linewidths=1.5,extent=extent)
ax.set_xlabel(r'$\theta_1$ [arcsec]',fontsize=20)
ax.set_ylabel(r'$\theta_2$ [arcsec]',fontsize=20)
# set ticks sizes
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
plt.tight_layout()
#
fig.savefig('mu_z3.png', dpi=300)
plt.show()

# write the detA map into a fits file
hdu = fits.PrimaryHDU(1.0/abs(detA),header=hdr)
# Correct the header update
hdu.header.update(wcs.to_header())
hdu.writeto('mu_elgordo_z3.fits',overwrite=True)



df.change_redshift(6.0)
kappa = df.ka
gamma1, gamma2 = df.g1, df.g2
detA = (1.0-kappa)**2 - gamma1**2 - gamma2**2
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(1.0/abs(detA),vmax=50.0,vmin=1.0,origin='lower',extent=extent)
ax.contour(detA, levels=[0.0], colors='red', linewidths=1.5,extent=extent)
ax.set_xlabel(r'$\theta_1$ [arcsec]',fontsize=20)
ax.set_ylabel(r'$\theta_2$ [arcsec]',fontsize=20)
# set ticks sizes
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
plt.tight_layout()
#
fig.savefig('mu_z6.png', dpi=300)
plt.show()

# write the detA map into a fits file
hdu = fits.PrimaryHDU(1.0/abs(detA),header=hdr)
# Correct the header update
hdu.header.update(wcs.to_header())
hdu.writeto('mu_elgordo_z6.fits',overwrite=True)

print (a1,a2)
beta1 = theta1 - a1
beta2 = theta2 - a2
#print (beta1,beta2)

zs = np.linspace(1.0,9.0, 10)
ggsl_list = []
for z in zs:
    df.change_redshift(z)

    tl = df.tancl()
    cl = df.getCaustics(tl)
    ggsl_cs = df.ggslCrossSection(clt=tl,minsize=0.5,maxsize=3.0)
    ggsl_list.append(ggsl_cs)

#fig,ax = plt.subplots(1,1,figsize=(10,10))
#ax.plot(zs,ggsl_list)
#ax.set_yscale('log')
#plt.show()

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(df.ka,vmax=2.0,vmin=0.0,origin='lower',extent=extent)
for t in tl:
    x,y = df.getCritPoints(t)
    ax.plot(x,y,'-',color='white')

for c in cl:
    x,y = df.getCausticPoints(c)
    ax.plot(x,y,'-',color='red')
plt.show()

kwargs = {
    'n': 1.0,
    'q': 1.0,
    'pa': np.pi/7.0,
    're': 1.0,
    'flux': 100.0,
    'zs': 3.0,
    'ys1': 27.88,
    'ys2': -45.16,
}

Npix = 2000#6666
se = sersic(size=fov_side, Npix=Npix, gl=df, save_unlensed=True, **kwargs)
ps = pointsrc(size=fov_side, Npix=Npix, gl=df, **kwargs)
xi,yi, mui = ps.find_images()

#se1 = sersic(size=200.0, Npix=2000, gl=df, ys1=5.8, ys2=0.0, **kwargs)

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(se.image,origin='lower',extent=fov)
ax.plot(xi,yi,'o',color='red')
plt.show()

# change wcs according to the new image size
factor_naxis = Npix / df.angx.shape[0]
wcs.wcs.cdelt /= factor_naxis
wcs.wcs.crpix = [Npix/2.0+0.5, Npix/2.0+0.5]



# save image into a fits file, adding wcs from the deflection angle map
header = wcs.to_header()
header['zs'] = kwargs['zs']
header['n'] = kwargs['n']
header['ys1'] = kwargs['ys1']
header['ys2'] = kwargs['ys2']
header['flux'] = kwargs['flux']
header['re'] = kwargs['re']
header['q'] = kwargs['q']
header['pa'] = kwargs['pa']

hdu = fits.PrimaryHDU(se.image,header=header)
# Correct the header update
hdu.header.update(wcs.to_header())
hdu.writeto('lensed_arc_0.fits',overwrite=True)

hdu = fits.PrimaryHDU(se.image_unlensed,header=header)
# Correct the header update
hdu.header.update(wcs.to_header())
hdu.writeto('unlensed_0.fits',overwrite=True)


print ('##############')
print (wcs)

ob = observation(size=fov_side, Npix=Npix, zp=23.9, mlim=24.5, rap=1.0, sn=5.0)
# if we want to simulate the observation with a particular instrument, we should convolve the image
# with the instrument PSF.
### CONVOLUTION HERE

# add noise:
noise=ob.makeNoise(se.image)
image = se.image + noise
hdu = fits.PrimaryHDU(image,header=header)
hdu.header.update(wcs.to_header())
hdu.writeto('lensed_arc_0_with_noise.fits',overwrite=True)





