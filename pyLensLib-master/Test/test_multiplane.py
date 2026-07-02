from pyLensLib.multiplane import *
from pyLensLib.piemd import piemd
from pyLensLib.deflector import deflector
import astropy.io.fits as pyfits


# test with existing lensplanes

"""
the lens planes are stored in fits files. Each fits file has to contain
certain keywords in the header. The keywords are:
            self.z = myhd[0].header['ZLENS']
            self.xmin = myhd[0].header['XMIN']
            self.xmax = myhd[0].header['XMAX']
            self.ymin = myhd[0].header['YMIN']
            self.ymax = myhd[0].header['YMAX']
            omega = myhd[0].header['OMEGA']
            hubble = myhd[0].header['H']
Two separate hdus contain the maps of the deflection angle components
"""

listplane=['lensplane_z=0.25.fits',
           'deflAnglesAres.fits',
           'lensplane_z=0.75.fits']

"""
create a list of lensplane objects
"""
lp=[]
for lenspl in listplane:
    lp.append(lensplane(lenspl))

"""
set the source redshift of the multiple plane deflector
"""
zs=2.0

"""
create a multiplane object
"""
mp=multiplane(lp,zs)
print ('Test 1: build a multiplane object from a list of files [OK]')

"""
In the second test, we plot the effective convergence maps at three different source redshifts
"""

"""
Create a mesh on the lens plane
"""
theta=np.linspace(-150.,150.,1000)
theta1,theta2=np.meshgrid(theta,theta)
def convergence(a1,a2,px):
    """
    function to compute the convergence from the deflection angles
    :param a1: def. angle first component
    :param a2: def. angle second component
    :param px: pixel scale (must be in the same units of a1 and a2!)
    :return: convergence map
    """

    a12,a11=np.gradient(a1/px)
    a22,a21=np.gradient(a2/px)
    ka=0.5*(a11+a22)
    g1=0.5*(a11-a22)
    g2=a21
    return (ka,g1,g2)


px=theta[1]-theta[0]

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'cubehelix_r'
fig,ax=plt.subplots(1,3,figsize=(18,8),sharey=True)
ip=0
"""
We plot the convergence for three source redshifts
"""
zs_arr=[0.4,0.65,3.0]
for zs in zs_arr:
    """
    Build the multiplane using the list of lensplanes and the source redshift
    """
    mp=multiplane(lp,zs)

    """
    compute the effective convergence 
    """
    beta1_,beta2_=mp.raytrace(np.deg2rad(theta1/3600.0),np.deg2rad(theta2/3600.0))
    beta1_=np.rad2deg(beta1_)*3600.0
    beta2_=np.rad2deg(beta2_)*3600.0
    a1=theta1-beta1_
    a2=theta2-beta2_
    ka,g1,g2=convergence(a1,a2,px)
    ax[ip].imshow(ka,origin='lower',extent=[theta.min(),theta.max(),theta.min(),theta.max()],vmax=ka.max()*0.15)
    ax[ip].contour((1.-ka)-np.sqrt(g1**2+g2**2),linestyles='-',
                   levels=[0.0],colors='yellow',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
    ax[ip].contour((1.-ka)+np.sqrt(g1**2+g2**2),linestyles='-',
                   levels=[0.0],colors='yellow',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
    ax[ip].set_xlim([theta.min(),theta.max()])
    ax[ip].set_ylim([theta.min(),theta.max()])
    ax[ip].set_aspect('equal')
    ip+=1

label_size=22
for i in range(3):
    ax[i].set_aspect('equal')
    ax[i].set_xlabel(r'$\theta_1$ [arcsec]',fontsize=label_size)
    ax[i].text(-130,90,r'$z_s$='+str(zs_arr[i]),fontsize=label_size)
ax[0].set_ylabel(r'$\theta_2$ [arcsec]',fontsize=label_size)
fig.tight_layout()
fig.savefig('effconv_planes.pdf',bbox_inches='tight',dpi=150)
for i in range(3):
    ax[i].set_aspect('equal')
    ax[i].set_xlabel(r'$\theta_1$ [arcsec]',fontsize=label_size)
    ax[i].text(-130,90,r'$z_s$='+str(zs_arr[i]),fontsize=label_size)
ax[0].set_ylabel(r'$\theta_2$ [arcsec]',fontsize=label_size)
fig.tight_layout()
fig.savefig('effconv_planes.pdf',bbox_inches='tight',dpi=150)

print ('Test 2: effective convergence from lensplanes read from fits files [see effconv_planes.pdf]')


# test with existing lensplanes:

"""
In this test, we create the lens planes on the fly using a
combination of piemds and deflector objects
"""
def gen_random_uniform(sigma_min=0.0,sigma_max=1.0):
    sigma=(sigma_max-sigma_min)*(np.random.random_sample((ngal,)))+sigma_min
    return sigma

zl=[0.25,0.5,0.75]
theta=np.linspace(-150.,150.,2048)
fov=300
ngal = int(3*(fov/60)**2)
rt=0.1
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

lp=[]
iplane=0
for zl_ in zl:
    iplane+=1
    gl=[]
    if zl_< 0.5 or zl_ > 0.5:
        sigma_max = 200.0
        sigma_min = 10.0
        sigma = gen_random_uniform(sigma_min, sigma_max)

        pa_max = np.pi
        pa_min = 0.0
        pa = gen_random_uniform(pa_min, pa_max)

        q_max = 1.0
        q_min = 0.5
        q = gen_random_uniform(q_min, q_max)

        x_max = fov / 2.
        x_min = -x_max
        x = gen_random_uniform(x_min, x_max)

        y_max = fov / 2.
        y_min = -fov / 2.
        y = gen_random_uniform(y_min, y_max)
        #print (ngal,x.min(),x.max(),y.min(),y.max())

        rt_as = np.rad2deg(rt / cosmo.angular_diameter_distance(zl_).value) * 3600.0
        theta = np.linspace(-fov / 2. , fov / 2., 1000)
        for i in range(ngal):
            kwargs = {'zl': zl_,
                      'zs': zs,
                      'sigma0': sigma[i],
                      'q': q[i],
                      'pa': pa[i],
                      'theta_c': 0.001,
                      'theta_t': rt_as,
                      'x1': x[i],
                      'x2': y[i]}

            dpie = piemd(cosmo, **kwargs)
            dpie.setGrid(theta)
            gl.append(dpie)
        lp.append(lensplane(gl,fromfits=False))
    else:
        myhd1 = pyfits.open("deflAnglesAres.fits")
        a1 = myhd1[0].data
        a2 = myhd1[1].data
        xmin = myhd1[0].header['XMIN']
        xmax = myhd1[0].header['XMAX']
        theta=np.linspace(xmin,xmax,a1.shape[0])
        zl_ = myhd1[0].header['ZLENS']
        dls=cosmo.angular_diameter_distance_z1z2(zl_,1.0).value
        ds=cosmo.angular_diameter_distance(1.0).value
        kwargs = {'zl': zl_, 'zs': 1.0}
        df = deflector(cosmo, angx=a1*dls/ds, angy=a2*dls/ds, **kwargs)
        df.setGrid(theta)
        lp.append(lensplane([df],fromfits=False))

fig,ax=plt.subplots(1,3,figsize=(18,8),sharey=True)
ip=0
zs_arr=[0.4,0.65,3.0]
for zs in zs_arr:
    mp=multiplane(lp,zs)
    beta1_,beta2_=mp.raytrace(np.deg2rad(theta1/3600.0),np.deg2rad(theta2/3600.0))
    beta1_=np.rad2deg(beta1_)*3600.0
    beta2_=np.rad2deg(beta2_)*3600.0
    a1=theta1-beta1_
    a2=theta2-beta2_
    ka,g1,g2=convergence(a1,a2,px)
    ax[ip].imshow(ka,origin='lower',extent=[theta.min(),theta.max(),theta.min(),theta.max()],vmax=ka.max()*0.15)
    ax[ip].contour((1.-ka)-np.sqrt(g1**2+g2**2),linestyles='-',
                   levels=[0.0],colors='yellow',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
    ax[ip].contour((1.-ka)+np.sqrt(g1**2+g2**2),linestyles='-',
                   levels=[0.0],colors='yellow',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
    ax[ip].set_xlim([theta.min(),theta.max()])
    ax[ip].set_ylim([theta.min(),theta.max()])
    ax[ip].set_aspect('equal')
    ip+=1

label_size=22
for i in range(3):
    ax[i].set_aspect('equal')
    ax[i].set_xlabel(r'$\theta_1$ [arcsec]',fontsize=label_size)
    ax[i].text(-130,90,r'$z_s$='+str(zs_arr[i]),fontsize=label_size)
ax[0].set_ylabel(r'$\theta_2$ [arcsec]',fontsize=label_size)
fig.tight_layout()
fig.savefig('effconv_planes.pdf',bbox_inches='tight',dpi=150)
for i in range(3):
    ax[i].set_aspect('equal')
    ax[i].set_xlabel(r'$\theta_1$ [arcsec]',fontsize=label_size)
    ax[i].text(-130,90,r'$z_s$='+str(zs_arr[i]),fontsize=label_size)
ax[0].set_ylabel(r'$\theta_2$ [arcsec]',fontsize=label_size)
fig.tight_layout()
fig.savefig('effconv_planes2.pdf',bbox_inches='tight',dpi=150)

print ('Test 3: effective convergence from lensplanes created on-th-fly [see effconv_planes2.pdf]')
