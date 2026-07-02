
import matplotlib.pyplot as plt
from pyLensLib.piemd import piemd
from pyLensLib.nfwell import nfwell
import numpy as np
from astropy import constants as const
import astropy.units as units
from astropy.cosmology import FlatLambdaCDM

import matplotlib

matplotlib.use('TkAgg')

co = FlatLambdaCDM(H0=70.0, Om0=0.3)
GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value

def bergamini_rcut(sigma):
    rcut=10.1*(sigma/220.0)**2.43
    return rcut

def bergamini_sigma(M):
    fact=3.57e11#np.pi*10.1*220.0**2.43/GG/1000.0
    sigma=220.0*(M/fact)**(1.0/4.43)
    return sigma

def rcut_from_mass_sigma(M,sigma):
    rcut = M/np.pi/sigma**2*GG
    return rcut


# Toy model 1: we fix the total mass and compare the einstein radii and the
# cross sections of a galaxy embedded into an NFW dark matter halo, placing the
# galaxy just outside the Einstein radius. In the simulation case we fix the
# velocity dispersion to that of the observed case

mass = 1e11
zl=0.4
zs=6.0

# need to convert rcut to arcsec
dl=co.angular_diameter_distance(zl)
conv_fact = np.rad2deg(1 /dl.value)*3600.0/1000.0

# set up the model consistently to observations (using the scaling relations of B+19
theta_c=0.1 # one pixel resolution
sigma_b = bergamini_sigma(mass)
rcut_b = bergamini_rcut(sigma_b)
rcut_b = np.rad2deg(rcut_b/dl.value/1000.0)*3600.0

kwargs_b = {'zl': zl,'zs': zs,'sigma0': sigma_b,'theta_c': theta_c,'theta_t': rcut_b}
p_b = piemd(co,**kwargs_b)


# for the simulation case, we introduce a core whose size is consistent with softening
theta_c_kpc=2.0 # kpc
theta_c_kpc_=np.rad2deg(theta_c_kpc/dl.value/1000.0)*3600.0 # arcsec
sigma_sim = sigma_b # 30% lower than bergamini
rcut_sim = rcut_from_mass_sigma(mass/1.2,sigma_sim)
rcut_sim = np.rad2deg(rcut_sim/dl.value)*3600.0

kwargs_sim = {'zl': zl,'zs': zs,'sigma0': sigma_sim,'theta_c': theta_c_kpc_,'theta_t': rcut_sim}
p_sim = piemd(co,**kwargs_sim)

# host halo (NFW) with an offset of 23 arcsec
kwargs_host = {'zl': zl, 'zs': zs, 'mass': 1e15, 'conc': 4.0, 'x1': 23.0, 'x2':0.0}
nfw=nfwell(co,**kwargs_host)

# combine galaxy and host halo
theta=np.linspace(-25.0,25.0,2000)
res=theta[1]-theta[0]
p_b.setGrid(theta=theta)
p_sim.setGrid(theta=theta)
nfw.setGrid(theta=theta)
p_b.combinewith(nfw)
p_sim.combinewith(nfw)

# caustics and critical lines
cl_b=p_b.tancl()
cl_sim=p_sim.tancl()
tcl_b=p_b.getCaustics(cl_b)
tcl_sim=p_b.getCaustics(cl_sim)

# plot of convergence, critical lines, and caustics
fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(p_sim.ka,origin='lower',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
ax.contour(1.0-p_sim.ka-np.sqrt(p_sim.g1**2+p_sim.g2**2),levels=[0.0],colors='white',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
ax.contour(1.0-p_b.ka-np.sqrt(p_b.g1**2+p_b.g2**2),levels=[0.0],colors='white',linestyles='--',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
for cau in tcl_b:
    x,y=p_b.getCausticPoints(cau)
    ax.plot(x,y,'--',color='yellow')

for cau in tcl_sim:
    x,y=p_sim.getCausticPoints(cau)
    ax.plot(x,y,'--',color='yellow')

fig.savefig('subs_in_nfw_coreradius.png')
plt.show()

# plot circular velocity and size of the Einstein radius
r=np.logspace(-4,np.log10(30.0/1000.0),100)
r_as=np.rad2deg(r/dl.value)*3600.0
vc_b = p_b.vcirc(r)
vc_sim = p_sim.vcirc(r)
m_b = p_b.m3Dr(r)
m_sim = p_sim.m3Dr(r)

print ('caustics area: %8.4f %8.4f' % (tcl_b[1].getArea()*p_b.grid_pixel**2,
                                       tcl_sim[1].getArea()*p_sim.grid_pixel**2))

theta_e_b_=p_b.thetaEv(cl_b)
theta_e_sim_=p_sim.thetaEv(cl_sim)
theta_e_b=theta_e_b_[-1]/conv_fact#*res
theta_e_sim=theta_e_sim_[-1]/conv_fact#*res


print ("ERs (arcsec): %8.4f %8.4f" % (theta_e_b_[-1],theta_e_sim_[-1]))
print ("ERs: %8.4f %8.4f" % (theta_e_b,theta_e_sim))
print ('Vmax: %10.4f %10.4f %10.4f' %  (np.max(vc_b),np.max(vc_sim),np.max(vc_b)/np.max(vc_sim)))
print ('M30kpc: %10.4e %10.4e %10.4e' %  (np.max(m_b),np.max(m_sim),np.max(m_b)/np.max(m_sim)))

fig,ax=plt.subplots(1,2,figsize=(18,10))
ax[0].plot(r*1000.,vc_b,'--')
ax[0].plot(r*1000.,vc_sim,'-')
y=np.ones(len(r))*1e12
y_e=[1,400]
x_e_b=[theta_e_b,theta_e_b]
x_e_sim=[theta_e_sim, theta_e_sim]
ax[0].plot(x_e_b,y_e,'--',color='red')
ax[0].plot(x_e_sim,y_e,'-',color='red')
#ax.plot(r,y,':')
ax[1].plot(r*1000.,p_b.m3Dr(r),'--')
ax[1].plot(r*1000.,p_sim.m3Dr(r),'-')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_xscale('log')
ax[1].set_xscale('log')
fsize=22
ax[0].xaxis.set_tick_params(labelsize=fsize)
ax[0].yaxis.set_tick_params(labelsize=fsize)
ax[1].xaxis.set_tick_params(labelsize=fsize)
ax[1].yaxis.set_tick_params(labelsize=fsize)
ax[0].set_xlabel(r'$r$ [kpc]',fontsize=fsize)
ax[1].set_xlabel(r'$r$ [kpc]',fontsize=fsize)
ax[0].set_ylabel(r'$V_{circ}$ [km/s]',fontsize=fsize)
ax[1].set_ylabel(r'$M(r)$ [$M_{\odot}$]',fontsize=fsize)
fig.tight_layout()
fig.savefig('vcirc_m3d_corerad.png')

plt.show()


# Toy model 2: we fix the total mass and compare the einstein radii and the
# cross sections of a galaxy embedded into an NFW dark matter halo. The velocity dispersion
# in the simulation case is lower than in the observed case

mass = 1e11
zl=0.4
zs=6.0

# need to convert rcut to arcsec
dl=co.angular_diameter_distance(zl)
conv_fact = np.rad2deg(1 /dl.value)*3600.0/1000.0

# set up the model consistently to observations (using the scaling relations of B+19
theta_c=0.1 # one pixel resolution
sigma_b = bergamini_sigma(mass)
rcut_b = bergamini_rcut(sigma_b)
rcut_b = np.rad2deg(rcut_b/dl.value/1000.0)*3600.0

kwargs_b = {'zl': zl,'zs': zs,'sigma0': sigma_b,'theta_c': theta_c,'theta_t': rcut_b}
p_b = piemd(co,**kwargs_b)


# for the simulation case, we reduce the sigma by 30%
sigma_sim = sigma_b/1.5 # 30% lower than bergamini
rcut_sim = rcut_from_mass_sigma(mass,sigma_sim)
rcut_sim = np.rad2deg(rcut_sim/dl.value)*3600.0

kwargs_sim = {'zl': zl,'zs': zs,'sigma0': sigma_sim,'theta_c': theta_c,'theta_t': rcut_sim}
p_sim = piemd(co,**kwargs_sim)

# host halo (NFW) with an offset of 23 arcsec
kwargs_host = {'zl': zl, 'zs': zs, 'mass': 1e15, 'conc': 4.0, 'x1': 23.0, 'x2':0.0}
nfw=nfwell(co,**kwargs_host)

# combine galaxy and host halo
theta=np.linspace(-25.0,25.0,2000)
res=theta[1]-theta[0]
p_b.setGrid(theta=theta)
p_sim.setGrid(theta=theta)
nfw.setGrid(theta=theta)
p_b.combinewith(nfw)
p_sim.combinewith(nfw)

# caustics and critical lines
cl_b=p_b.tancl()
cl_sim=p_sim.tancl()
tcl_b=p_b.getCaustics(cl_b)
tcl_sim=p_b.getCaustics(cl_sim)

# plot of convergence, critical lines, and caustics
fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(p_sim.ka,origin='lower',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
ax.contour(1.0-p_sim.ka-np.sqrt(p_sim.g1**2+p_sim.g2**2),levels=[0.0],colors='white',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
ax.contour(1.0-p_b.ka-np.sqrt(p_b.g1**2+p_b.g2**2),levels=[0.0],colors='white',linestyles='--',extent=[theta.min(),theta.max(),theta.min(),theta.max()])
for cau in tcl_b:
    x,y=p_b.getCausticPoints(cau)
    ax.plot(x,y,'--',color='yellow')

for cau in tcl_sim:
    x,y=p_sim.getCausticPoints(cau)
    ax.plot(x,y,'--',color='yellow')

fig.savefig('subs_in_nfw_coreradius.png')
plt.show()

# plot circular velocity and size of the Einstein radius
r=np.logspace(-4,np.log10(30.0/1000.0),100)
r_as=np.rad2deg(r/dl.value)*3600.0
vc_b = p_b.vcirc(r)
vc_sim = p_sim.vcirc(r)
m_b = p_b.m3Dr(r)
m_sim = p_sim.m3Dr(r)

print ('caustics area: %8.4f %8.4f' % (tcl_b[1].getArea()*p_b.grid_pixel**2,
                                       tcl_sim[1].getArea()*p_sim.grid_pixel**2))

theta_e_b_=p_b.thetaEv(cl_b)
theta_e_sim_=p_sim.thetaEv(cl_sim)
theta_e_b=theta_e_b_[-1]/conv_fact#*res
theta_e_sim=theta_e_sim_[-1]/conv_fact#*res


print ("ERs (arcsec): %8.4f %8.4f" % (theta_e_b_[-1],theta_e_sim_[-1]))
print ("ERs: %8.4f %8.4f" % (theta_e_b,theta_e_sim))
print ('Vmax: %10.4f %10.4f %10.4f' %  (np.max(vc_b),np.max(vc_sim),np.max(vc_b)/np.max(vc_sim)))
print ('M30kpc: %10.4e %10.4e %10.4e' %  (np.max(m_b),np.max(m_sim),np.max(m_b)/np.max(m_sim)))

fig,ax=plt.subplots(1,2,figsize=(18,10))
ax[0].plot(r*1000.,vc_b,'--')
ax[0].plot(r*1000.,vc_sim,'-')
y=np.ones(len(r))*1e12
y_e=[1,400]
x_e_b=[theta_e_b,theta_e_b]
x_e_sim=[theta_e_sim, theta_e_sim]
ax[0].plot(x_e_b,y_e,'--',color='red')
ax[0].plot(x_e_sim,y_e,'-',color='red')
#ax.plot(r,y,':')
ax[1].plot(r*1000.,p_b.m3Dr(r),'--')
ax[1].plot(r*1000.,p_sim.m3Dr(r),'-')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_xscale('log')
ax[1].set_xscale('log')
fsize=22
ax[0].xaxis.set_tick_params(labelsize=fsize)
ax[0].yaxis.set_tick_params(labelsize=fsize)
ax[1].xaxis.set_tick_params(labelsize=fsize)
ax[1].yaxis.set_tick_params(labelsize=fsize)
ax[0].set_xlabel(r'$r$ [kpc]',fontsize=fsize)
ax[1].set_xlabel(r'$r$ [kpc]',fontsize=fsize)
ax[0].set_ylabel(r'$V_{circ}$ [km/s]',fontsize=fsize)
ax[1].set_ylabel(r'$M(r)$ [$M_{\odot}$]',fontsize=fsize)
fig.tight_layout()
fig.savefig('vcirc_m3d_corerad.png')

plt.show()