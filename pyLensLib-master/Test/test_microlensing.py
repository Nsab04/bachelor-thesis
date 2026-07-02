from pyLensLib.pointsrc import pointsrc
from pyLensLib.observation import observation
from astropy.io import fits as fits
import matplotlib


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyLensLib.microlensing import *

ps=point_source()

# set up the lens model and the monitoring time

M_lens = 0.3  # solar masses
DL = 4.0  # kpc
DS = 8.0  # kpc
vel = 200  # km/s

# initialize the impact parameters
y0 = np.array([0.1])


# passage at the minimum distance from the lens
t0 = 365  # days

pl=point_lens(ps,mass=M_lens,dl=DL,ds=DS,t0=t0,y0=y0[0])

# compute the Einstein radius crossing time
t_einst = pl.EinsteinCrossTime() #EinsteinCrossTime(M_lens, DL, DS, vel).value  # days

t = t0 + np.linspace(-2, 2, 501) * t_einst.value

xx = [-2, 2]

fig, ax = plt.subplots(1, 2, figsize=(18, 8))
ax[1].set_ylim([1.0, 10.0])
ax[1].set_xlim([-2, 2])
ax[1].set_yscale('log')
ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_tick_params(labelsize=20)
ax[1].yaxis.set_tick_params(labelsize=20)
ax[1].set_yticks(np.arange(1, 11, 1.0))
ax[1].set_xlabel(r'$(t-t_0)/t_E$', fontsize=23)
ax[1].set_ylabel(r'$\mu(t)$', fontsize=23)

circle = plt.Circle((0, 0), 1, color='black', fill=False)
ax[0].set_xlim([-1.5, 1.5])
ax[0].set_ylim([-1.5, 1.5])
ax[0].add_artist(circle)
ax[0].plot([0.0], [0.0], '*', markersize=20, color='red')
ax[0].xaxis.set_tick_params(labelsize=20)
ax[0].yaxis.set_tick_params(labelsize=20)

from matplotlib.pyplot import cm

color = iter(cm.rainbow(np.linspace(0, 1, y0.size)))

# loop over the impact parameters and plot the light curves
for i in range(y0.size):
    c = next(color)
    mut = -pl.mum(t) + pl.mup(t)
    # mut_func(t_einst, y0[i], t0, t)
    ax[1].plot((t - t0) / t_einst.value, mut, '-', color=c)
    yy = [y0[i], y0[i]]
    ax[0].plot(xx, yy, '--', color=c, lw=2)

print (mut)
ax[0].set_xlabel(r'$y_1$', fontsize=23)
ax[0].set_ylabel(r'$y_2$', fontsize=23)
plt.show()

##################

nt=15 #15
t_sample = t0 + np.linspace(-1, 1, nt) * t_einst.value
mut_sample = -pl.mum(t_sample) + pl.mup(t_sample)# mut_func(t_einst, y0[0], t0, t_sample)

sz=0.5
npix=int(sz/0.01)+1
zp=24.0
ob = observation(size=sz, Npix=npix, zp=zp, texp=2000, bkg=23.5)


mag_star=22.0
#fl_star = mut_sample*ob.mag2counts(mag_star)

hdul = fits.open('/Users/massimo/share/astro/PSF/HST_ACS_WFC_PSF/HST_ACS_WFC_F814W_PSF00.fits', memmap=False)
psf_image = hdul[0].data

#fig,ax=plt.subplots(1,nt,figsize=(15,3),sharey=True)
plt.style.use('dark_background')
fig=plt.figure(figsize=(15,15),dpi=600)

grid = plt.GridSpec(nt, nt, hspace=0.0, wspace=0.2)
ax_main = fig.add_subplot(grid[1:nt,0:nt])
ax=[]
for i in range(nt):
    ax.append(fig.add_subplot(grid[0,i],xticklabels=[], yticklabels=[]))

ax_main.plot((t - t0), mut, '-', color='white')
ax_main.xaxis.set_tick_params(labelsize=20)
ax_main.yaxis.set_tick_params(labelsize=20)

for i in range(nt):
    fl_star = mut_sample[i] * ob.mag2counts(mag_star)
    kwargs = {
        'ys1': 0.0,
        'ys2': 0.0,
        'flux': fl_star,
        'zs': 1.0
    }

    star=pointsrc(size=sz,Npix=npix,gl=None,**kwargs)
    image=star.image
    image_conv = ob.convolve_psf(image, psf_image, 0.0495)
    noise = ob.makeNoise(image_conv)

    ax[i].imshow((image_conv+noise) , origin='low', extent=[-sz / 2., sz / 2., -sz / 2., sz / 2.],vmax=0.5,cmap='magma')
#plt.tight_layout()
ax_main.set_xlabel('Tempo [giorni]',fontsize=23)
ax_main.set_ylabel('Amplificazione',fontsize=23)
plt.savefig('microlensing_esempio.png')
plt.show()


# a planetary microlensing event

#bl=binary_lens(ps,dl=5.0,m1=1.0,q=1.e-3,d=1.0,t0=365.0,y0=0.1,theta=0.0)
#bl=binary_lens(ps,m1=1.0,q=1.e-3,d=1.2,t0=0.0,y0=-0.3,theta=np.pi/4.)
q=1e-3
bl=binary_lens(m1=1.0*q,q=q,d=0.95,t0=0,y0=0.01,theta=0.45*np.pi,centre_m2=True)
x1,x2,xs1,xs2=bl.CritCau()

times=np.linspace(-1290,290,730)
#t = np.linspace(-290, 290, 501)

color = iter(cm.rainbow(np.linspace(0, 1, times.size)))

fig, ax = plt.subplots(1, 2, figsize=(18, 8))

for t in times:
    c = next(color)
    ys1, ys2 = bl.SourcePos(t)
    xi1, xi2 = bl.Images(ys1, ys2)
    ax[1].plot(ys1, ys2, '*', markersize=10, color=c)
    ax[0].plot(xi1, xi2, 'o', markersize=10, color=c)

ax[0].plot(x1, x2, ',', color='blue')
ax[1].plot(xs1, xs2, ',', color='blue')
#################################################################

z1 = bl.getPos()
z2 = -z1
ax[0].plot([z1.real], [z1.imag], '*', markersize=10, color='blue')
ax[0].plot([z2.real], [z2.imag], '*', markersize=10, color='red')

# set dimensions of image plane plotting area
xmin = np.amin(x1)
xmax = np.amax(x1)
ymin = np.amin(x2)
ymax = np.amax(x2)
dim = [xmax - xmin, ymax - ymin]
if (dim[0] > dim[1]):
    side = dim[0] * 1.1
else:
    side = dim[1] * 1.1

xmin_ = 0.5 * (xmin + xmax) - 1.3 * side / 2.0
xmax_ = 0.5 * (xmin + xmax) + 1.3 * side / 2.0
ymin_ = 0.5 * (ymin + ymax) - 1.3 * side / 2.0
ymax_ = 0.5 * (ymin + ymax) + 1.3 * side / 2.0

ax[0].set_xlim([xmin_, xmax_])
ax[0].set_ylim([ymin_, ymax_])

# set dimensions of source plane plotting area
xmin = np.amin(xs1)
xmax = np.amax(xs1)
ymin = np.amin(xs2)
ymax = np.amax(xs2)
dim = [xmax - xmin, ymax - ymin]
if (dim[0] > dim[1]):
    side = dim[0] * 3.5
else:
    side = dim[1] * 3.5

xmin_ = 0.5 * (xmin + xmax) - side / 3.0
xmax_ = 0.5 * (xmin + xmax) + side / 3.0
ymin_ = 0.5 * (ymin + ymax) - side / 3.0
ymax_ = 0.5 * (ymin + ymax) + side / 3.0

ax[1].set_xlim([xmin_, xmax_])
ax[1].set_ylim([ymin_, ymax_])

ax[0].xaxis.set_tick_params(labelsize=20)
ax[0].yaxis.set_tick_params(labelsize=20)
ax[1].xaxis.set_tick_params(labelsize=20)
ax[1].yaxis.set_tick_params(labelsize=20)

ax[0].set_xlabel('$x_1$', fontsize=20)
ax[0].set_ylabel('$x_2$', fontsize=20)
ax[1].set_xlabel('$y_1$', fontsize=20)
ax[1].set_ylabel('$y_2$', fontsize=20)

fig.savefig('bl_multima.png')
plt.show()

p,mut=bl.LightCurve(times)







t_einst=bl.gettE()
t0=0.0
#t = t0 + np.linspace(-5, 5, 501) * t_einst.value
t_sample = t0 + np.linspace(-1, 1, nt) * t_einst.value
p_sample,mut_sample=bl.LightCurve(t_sample)
#p,mut=bl.LightCurve(t)

plt.style.use('dark_background')
fig=plt.figure(figsize=(15,15),dpi=600)

grid = plt.GridSpec(nt, nt, hspace=0.0, wspace=0.2)
ax_main = fig.add_subplot(grid[1:nt,0:nt])
ax=[]
for i in range(nt):
    ax.append(fig.add_subplot(grid[0,i],xticklabels=[], yticklabels=[]))

ax_main.plot(p*t_einst.value, mut, '-', color='white')
ax_main.xaxis.set_tick_params(labelsize=20)
ax_main.yaxis.set_tick_params(labelsize=20)

for i in range(nt):
    fl_star = mut_sample[i] * ob.mag2counts(mag_star)
    kwargs = {
        'ys1': 0.0,
        'ys2': 0.0,
        'flux': fl_star,
        'zs': 1.0
    }

    star=pointsrc(size=sz,Npix=npix,gl=None,**kwargs)
    image=star.image
    image_conv = ob.convolve_psf(image, psf_image, 0.0495)
    noise = ob.makeNoise(image_conv)

    ax[i].imshow((image_conv+noise) , origin='low', extent=[-sz / 2., sz / 2., -sz / 2., sz / 2.],vmax=0.5,cmap='magma')
#plt.tight_layout()
ax_main.set_xlabel('Tempo [giorni]',fontsize=23)
ax_main.set_ylabel('Amplificazione',fontsize=23)
plt.savefig('planet_microlensing_esempio.png')
plt.show()
