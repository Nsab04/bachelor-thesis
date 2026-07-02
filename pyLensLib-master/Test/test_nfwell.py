from pyLensLib.nfwell import nfwell
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import matplotlib.pyplot as plt

co = FlatLambdaCDM(H0=70.0, Om0=0.3)

zl=0.4
zs=1.2
mass=1.5e12
x=0.0
y=0.0

kwargs =  {'zl': zl, 'zs': zs, 'mass': mass, 'conc': 7.5, 'x1': x, 'x2': y, 'q': 0.5, 'pa': 0.0}
ne = nfwell(co,**kwargs)

npix = 5000
fov = 10.0
theta = np.linspace(-fov/2.,fov/2.,npix)
ne.setGrid(theta,theta)

tl = ne.tancl()
print (len(tl))
print (ne.thetaE())
exit()

a1 = ne.a1
a2 = ne.a2
k = ne.ka

a = np.sqrt(a1**2 + a2**2)
r = np.sqrt(ne.theta1**2 + ne.theta2**2)
k_nfw = ne.nfwkappa(r)
##ax.plot(r,a_gnfw,',',color='blue')
#ax.plot(r,(k-k_nfw)/k_nfw,',',color='blue')
krat = (k-k_nfw)/k_nfw
print(krat.min(),krat.max(),krat.mean(),krat.std())
#plt.show()

# plot two maps of the convergence using the functions kappa and kappa_exact of the nfwell class
# and compare them

kappa_exact = ne.kappa_exact(ne.theta1,ne.theta2)
ratio = (ne.ka-kappa_exact)/kappa_exact
print(ratio.min(),ratio.max(),ratio.mean(),ratio.std())

showFigure = True
if showFigure:
    fig,ax = plt.subplots(1,3,figsize=(24,10))
    ax[0].imshow(ne.ka)
    ax[0].contour(ne.ka,levels=[0.1,0.5,1.0,1.5,2.0],colors='white')

    ax[1].imshow(kappa_exact)
    ax[1].contour(kappa_exact,levels=[0.1,0.5,1.0,1.5,2.0],colors='white')
    ax[1].contour(ne.ka,levels=[0.1,0.5,1.0,1.5,2.0],linestyles='--',colors='yellow')
    #ax[2].imshow((ne.ka-kappa_exact)/kappa_exact, cmap='seismic',vmin=-0.1,vmax=0.1)


    # draw a profile of the difference between the two maps
    ax[2].plot(np.linspace(0,fov/2.,npix),ratio[npix//2,:],color='black')
    ax[2].plot(np.linspace(0,fov/2.,npix),ratio[:,npix//2],color='red')
    #ax[2].plot(r,krat,',',color='blue')
    plt.show()

