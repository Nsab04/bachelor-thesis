from pyLensLib.piemd import piemd
from pyLensLib.sersic import sersic
from pyLensLib.maputils import map_obj, contour_fit, image_fit
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
import astropy.units as units

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

kwargs = {'zl': 0.5,
          'zs': 1.0,
          'sigma0': 1200.0,
          'q': 0.8,
          'pa': -np.pi / 4.0,
          'theta_c': 2.0,
          'theta_t': 1000.0,
          'x1': 0.0,
          'x2': 0.0}


class nfwcirc(object):
    """
    class circular NFW lens
    """

    def __init__(self, co, zl=0.3, zs=2.0, c200=4.0, m200=1e15):  # , rs=0.3,rhos=1e15):
        """
        Initialize the nfwcirc object
        """
        self.co = co  # cosmological model
        self.zl = zl  # lens redshift
        self.zs = zs  # source redshift
        self.m200 = m200  # Mass of the lens
        self.c200 = c200  # concentration of the lens

        # compute rhos
        self.rhos = 200. / 3. * (self.co.critical_density(self.zl).to('Msun/Mpc3')) * self.c200 ** 3 / \
                    (np.log(1. + self.c200) - self.c200 / (1.0 + self.c200))

        # compute r200 and rs
        f200 = 4. / 3. * np.pi * 200. * self.co.critical_density(self.zl).to('Msun/Mpc3').value
        self.r200 = (self.m200 / f200) ** (1. / 3.)
        self.rs = self.r200 / self.c200

        # compute the angular diameter distances:
        self.dl = self.co.angular_diameter_distance(self.zl)
        self.ds = self.co.angular_diameter_distance(self.zs)
        self.dls = self.co.angular_diameter_distance_z1z2(self.zl, self.zs)

        # surface critical density:
        self.sc = self.sigma_crit()

        # convergence scale:
        self.ks = self.rhos.value * self.rs / self.sc

    def sigma_crit(self):
        """
        Function to compute the critical surface density
        """
        c2G = (const.c ** 2 / const.G).to(units.Msun / units.Mpc)
        factor = c2G / (4 * np.pi)
        return (factor * (self.ds / (self.dl * self.dls))).value

    def kappap(self, r):
        """
        Convergence at radius r [Mpc].
        """
        x = r / self.rs
        fx = np.piecewise(x, [x > 1., x < 1., x == 1.],
                          [lambda x: (1 - (2.0 / np.sqrt(x * x - 1.) *
                                           np.arctan(np.sqrt((x - 1.) / (x + 1.))))) / (x ** 2 - 1),
                           lambda x: (1 - (2.0 / np.sqrt(1. - x * x) *
                                           np.arctanh(np.sqrt((1. - x) / (1. + x))))) / (x ** 2 - 1),
                           1. / 3.])
        kappa = 2.0 * self.ks * fx
        return kappa

    def massp(self, r):
        """
        Dimensionless mass at radius r [Mpc]
        """
        x = r / self.rs
        fx = np.piecewise(x, [x > 1., x < 1., x == 1.],
                          [lambda x: (2.0 / np.sqrt(x * x - 1.) *
                                      np.arctan(np.sqrt((x - 1.) / (x + 1.)))),
                           lambda x: (2.0 / np.sqrt(1. - x * x) *
                                      np.arctanh(np.sqrt((1. - x) / (1. + x)))),
                           0])
        massp = 4.0 * self.ks * (np.log(x / 2.) + fx)
        return massp

    def shearp(self, r):
        """
        Shear at radius r [Mpc]
        """
        kp = self.kappap(r)
        mp = self.massp(r)
        x = r / self.rs
        gammap = -(kp - mp / x ** 2)
        return gammap

    def redshearp(self, r):
        """
        Reduced shear at radius r [Mpc]
        """
        redgammap = np.abs(self.shearp(r) / (1. - self.kappap(r)))
        return redgammap

def getImageEllipticity(img,fsize,f=0.05,plotax=None):
    m=map_obj(img)
    cnt = m.get_contours(lev=f)
    r=[]
    e=[]
    p=[]
    x=[]
    y=[]
    for c in cnt:
        c_fit = contour_fit(c)
        if (len(c_fit.x_list) > 5):
            c_fit.fitEllipse()
            center = c_fit.ellipse_center()
            phi = c_fit.ellipse_angle_of_rotation()
            axes = c_fit.ellipse_axis_length()
            R = np.arange(0, 2.0*np.pi, 0.01)
            a, b = axes
            print (a,b)
            if (plotax != None):
                #xx=c_fit.x_list*(fsize/(img.shape[0]-1))-fsize/2.0
                #yy=c_fit.y_list*(fsize/(img.shape[0]-1))-fsize/2.0
                #plotax.plot(xx,yy,'-',color='red')
                xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
                yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
                xx=(xx+0.5)*(fsize/(img.shape[0]))-fsize/2.0
                yy=(yy+0.5)*(fsize/(img.shape[0]))-fsize/2.0
                xc=(center[0]+0.5)*(fsize/(img.shape[0]))-fsize/2.0
                yc = (center[1] + 0.5) * (fsize / (img.shape[0])) - fsize / 2.0
                r.append(np.sqrt(xc*xc+yc*yc))
                x.append(xc)
                y.append(yc)
                e.append((a-b)/(a+b))
                p.append(phi)
                if (np.abs(xc)<fsize/2.-1.0) & (np.abs(yc)<fsize/2.-1.0):
                    plotax.plot(xx, yy, ':', color='red',zorder=10)
    return x,y,r,e,p

def getImageEllipticity2(img,fsize,f=0.05,plotax=None):
    imafit=image_fit(img,ith=f)
    if (len(imafit.img_sel)>0):
        e1, e2 = imafit.ellipticity()
        center = imafit.c
        a, b = imafit.axes()
        phi = 0.5 * np.arctan2(e2, e1)
        area_pix = np.pi * a * b
        a = a / np.sqrt(area_pix) * np.sqrt(len(imafit.img_sel))
        b = b / np.sqrt(area_pix) * np.sqrt(len(imafit.img_sel))
        print('moments', len(imafit.img_sel), a, b)
        R = np.arange(0, 2.0 * np.pi, 0.01)
        if (plotax != None):
            xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
            yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
            xx = (xx + 0.5) * (fsize / (img.shape[0])) - fsize / 2.0
            yy = (yy + 0.5) * (fsize / (img.shape[0])) - fsize / 2.0
            xc = (center[0] + 0.5) * (fsize / (img.shape[0])) - fsize / 2.0
            yc = (center[1] + 0.5) * (fsize / (img.shape[0])) - fsize / 2.0
            if (np.abs(xc) < fsize / 2. - 1.0) & (np.abs(yc) < fsize / 2. - 1.0):
                plotax.plot(xc, yc, '+', color='red')
                plotax.plot(xx, yy, ':', color='orange', zorder=12)
    return

dpie = piemd(co, **kwargs)

fsize_field = 500
fsize=600.0#140.0

theta = np.linspace(-fsize, fsize, 1000)
dpie.setGrid(theta)

ngal_side = 30
d = np.linspace(-fsize/2.0,fsize/2.0,ngal_side)
xg, yg = np.meshgrid(d,d)

fig,ax=plt.subplots(1,2,figsize=(20,13))
ii=0
xf=[]
yf=[]
rf=[]
ef=[]
pf=[]
for i in range(ngal_side):
    for j in range(ngal_side):
        print ('src ',i,j)
        kwargs = {
            'ys1': xg[i,j],
            'ys2': yg[i,j],
            'pa': 0.0,
            'flux': 100.,
            'zs': 1.0,
            're': 1.0
        }
        ps = sersic(size=fsize*1.2, Npix=1000, gl=dpie, **kwargs)
        print(ps.ys1,ps.ys2)
        if (ii == 0):
            image = ps.image.copy()
            ii += 1
            xe,ye,r,e,p=getImageEllipticity(ps.image, fsize=fsize*1.2, f=0.05, plotax=ax[1])
            #getImageEllipticity2(ps.image, fsize=fsize * 1.2, f=0.05, plotax=ax[1])
        else:
            image = image.copy() + ps.image.copy()
            xe,ye,r,e,p=getImageEllipticity(ps.image, fsize=fsize*1.2, f=0.05, plotax=ax[1])
            #getImageEllipticity2(ps.image, fsize=fsize * 1.2, f=0.05, plotax=ax[1])
        for k in range(len(r)):
            xf.append(xe[k])
            yf.append(ye[k])
            rf.append(r[k])
            ef.append(e[k])
            pf.append(p[k])
        print (ps.image.min(),ps.image.max())

ii=0
for i in range(ngal_side):
    for j in range(ngal_side):
        print ('src ',i,j)
        kwargs = {
            'ys1': xg[i,j],
            'ys2': yg[i,j],
            'pa': 0.0,
            'flux': 100.,
            'zs': 1.0
        }
        ps = sersic(size=fsize*1.2, Npix=1000, gl=None, **kwargs)
        print(ps.ys1,ps.ys2)
        if (ii == 0):
            image_unlensed = ps.image.copy()
            ii += 1
        else:
            image_unlensed = image_unlensed.copy() + ps.image.copy()
        print (ps.image.min(),ps.image.max())


ax[0].tick_params(
    which='both',
    bottom=False,
    top=False,
    labelbottom=False,
    left=False,
    right=False,
    labelleft=False
)
ax[1].tick_params(
    which='both',
    bottom=False,
    top=False,
    labelbottom=False,
    left=False,
    right=False,
    labelleft=False
)
ax[0].imshow(image_unlensed,cmap='gray_r',vmax=10.0e-1,vmin=0.5e-1,zorder=0,
             extent=[-fsize/2.0*1.2,fsize/2.0*1.2,-fsize/2.0*1.2,fsize/2.0*1.2],origin='lower')
ax[1].imshow(image,cmap='gray_r',vmax=10.0e-1,zorder=0,
             extent=[-fsize/2.0*1.2,fsize/2.0*1.2,-fsize/2.0*1.2,fsize/2.0*1.2],origin='lower')
tcl=dpie.tancl()
for c in tcl:
    x,y=dpie.getCritPoints(c)
    ax[1].plot(x,y,'-',color='blue')
    tcl_=np.sqrt(x*x+y*y).mean()
rcl=dpie.radcl()
for c in rcl:
    x,y=dpie.getCritPoints(c)
    ax[1].plot(x,y,'-',color='blue')
    rcl_ = np.sqrt(x * x + y * y).mean()

ax[1].set_xlim([-fsize/2.0*1.2,fsize/2.0*1.2])
ax[1].set_ylim([-fsize/2.0*1.2,fsize/2.0*1.2])
plt.tight_layout()
plt.savefig('sourcegrid_book.png')
plt.show()

fig,ax=plt.subplots(1,1,figsize=(10,10))
xf=np.array(xf)
yf=np.array(yf)
ef=np.array(ef)
rf=np.array(rf)
pf=np.array(pf)
e1=ef*np.cos(2.0*pf)
e2=ef*np.sin(2.0*pf)
varphi=np.arctan2(yf,xf)
er=np.abs((e1*np.cos(2*varphi)+e2*np.sin(2*varphi)))
ex=(e2*np.cos(2*varphi)-e1*np.sin(2*varphi))
ir=np.argsort(rf)
ga1_true,ga2_true=dpie.gamma(xf[ir],yf[ir])
ka_true=dpie.kappa(xf[ir],yf[ir])
g1_true=ga1_true/(1.0-ka_true)
g2_true=ga2_true/(1.0-ka_true)
gr=np.abs((g1_true*np.cos(2*varphi[ir])+g2_true*np.sin(2*varphi[ir])))
gx=np.abs((g2_true*np.cos(2*varphi[ir])-g1_true*np.sin(2*varphi[ir])))
ax.plot(rf,er,'o',label=r'$e_t$')
ax.plot(rf[ir],gr,'o',label=r'$g_t$',)
ax.plot(rf,ex,'x',label=r'$e_x$')
ax.plot(rf[ir],gx,'x',label=r'$g_x$')
ax.plot([tcl_,tcl_],[-0.02,0.5],'--',label='tan. c. l.',color='black')
ax.plot([rcl_,rcl_],[-0.02,0.5],'--',label='rad. c. l.',color='gray')
#ax.set_ylim([-0.1,1.0])

best_fit_c=5.18267193
best_fit_m=15.0010455
nc = nfwcirc(co,zl=0.5,zs=1.0,c200=best_fit_c,m200=10**best_fit_m)
r_as_model = np.linspace(20.0,480.0,50)
rm = r_as_model/180.0/3600.0*np.pi*nc.dl.value
gp = nc.redshearp(rm)
ax.plot(r_as_model,gp,':',color='blue',linewidth=2,label='best fit NFW')

ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
ax.set_xlabel(r'$\theta$ [arcsec]', fontsize=18)
ax.set_ylabel('ellipticity, shear', fontsize=18)
ax.legend(fontsize=18)
fig.savefig('dpie_shear_prof.png')
plt.show()

import pandas as pd
dict={'x': xf, 'y': yf, 'e1': e1, 'e2': e2, 'et': er, 'ex': ex,'g1': g1_true, 'g2': g2_true}
df=pd.DataFrame.from_dict(dict)
df.to_csv('dpie_sim_catalog.d',sep=' ')

