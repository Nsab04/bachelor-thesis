from pyLensLib.subfind import subfind
from pyLensLib.deflector import deflector
from pyLensLib.cluster import cluster
from pyLensLib.raymesh import raymesh

import numpy as np
import astropy.io.fits as fits

import matplotlib

matplotlib.use('TkAgg')

snapshot = '/Users/massimo/projects/ggsl/testing_codes/massmaps_python/BH_2015/snap_058'
subfind_file = "/Users/massimo/stiva/Dianoga/new_elena/Subfind_AGN/D1/sub_058.0"

sf=subfind(subfind_file)
xc,yc,zc=sf.gpos[0]
cl = cluster(snapshot=snapshot,parttype=[1],xc=xc,yc=yc,zc=zc)
import matplotlib.pyplot as plt

filealpha='/Users/massimo/stiva/Dianoga/new_elena/AGN_400_2048_2/lensing/1_z_058alpha.fits'
hdul=fits.open(filealpha)
a1=hdul[0].data
a2=hdul[1].data

kwargs_def={'zl': cl.zl, 'zs': 3.0}

dls=cl.co.angular_diameter_distance_z1z2(cl.zl,kwargs_def['zs'])
ds=cl.co.angular_diameter_distance(kwargs_def['zs'])
a1=a1*dls/ds
a2=a2*dls/ds

df=deflector(cl.co,angx=a1,angy=a2,**kwargs_def)
theta=np.linspace(-100,100,2048)
df.setGrid(theta=theta,compute_potential=False)

import pymupds


detA=(1.0-df.ka)**2-(df.g1**2+df.g2**2)
y1=df.theta1-df.a1+2*df.grid_pixel
y2=df.theta2-df.a2+2*df.grid_pixel

#plt.imshow(masked_deta,origin='low')
#plt.contour(1./np.abs(detA),levels=[0])
nrefin=4
rm=raymesh(nray_ini=detA.shape[0],rule=np.abs(1.0/detA),nref=nrefin)
#rm=raymesh(nray_ini=32,rule=np.abs(1.0/detA),nref=nrefin)
fig,ax=plt.subplots(1,2,figsize=(18,9))
for i in range(nrefin):
    ax[0].plot(rm.px[rm.refinement==i],rm.py[rm.refinement==i],',')
ax[0].set_aspect('equal')
y1s,y2s=df.mapCrit2Cau(rm.px,rm.py)
for i in range(nrefin):
    ax[1].plot(y1s[rm.refinement==i],y2s[rm.refinement==i],',')
ax[1].set_aspect('equal')
#plt.show()

mus=pymupds.mupds_triangle(y1-df.theta.min(),y2-df.theta.min(),df.grid_pixel,detA,nray=len(theta))
imm_sp=pymupds.sp_image_multiplicity_map(y1-df.theta.min(),y2-df.theta2.min(),df.grid_pixel,nray=len(theta))

ix= imm_sp == 3
print (ix.sum()*df.grid_pixel**2)
ix= imm_sp == 5
print (ix.sum()*df.grid_pixel**2)
ix= imm_sp == 7
print (ix.sum()*df.grid_pixel**2)

from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata
pos=np.stack((rm.px,rm.py),axis=-1)
imm_ip = map_coordinates(imm_sp, [[y1s], [y2s]], order=1, prefilter=True)
imm_ip = imm_ip.reshape(imm_ip.size)
grid_x, grid_y = np.mgrid[0:2047, 0:2047]
imm_ip_map = griddata(pos,imm_ip,(grid_x,grid_y),method='linear')

fig,ax=plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(imm_sp)
ax[1].imshow(imm_ip_map)
plt.show()

exit()


colo=['blue','red','green','orange','gray']
fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(np.log10(mus.T),origin='low',vmax=1.7)
inter,cs_mi=df.imageMolteplicity()
print (cs_mi)
"""
#print ([inter[i].area for i in range(len(inter))])
id=0
from shapely.ops import polygonize, unary_union
for i in range(len(inter)):
    print ('*',i)
    for geo in inter[i].geoms:
        if geo.geom_type == 'Polygon':
            x,y=geo.exterior.xy
            if (len(x)>1):
                ax.fill(x,y,color=colo[id],zorder=id)
    id+=1
"""
tl = df.tancl()
ctl = df.getCaustics(tl)
rl = df.radcl()
crl = df.getCaustics(rl)

fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(imm_sp.T,origin='low')
plt.show()

from shapely.ops import triangulate
import shapely.geometry as gp
for c in ctl:
    x,y=df.getCausticPoints(c,pixel_units=True)
    if (c.principale):
        #ax.plot(x,y,'-',color='black')
        print (c.geometria.geom_type,len(c.geometria))
        for i in range(len(c.geometria)):
            if c.geometria[i].geom_type == 'Polygon':
                x,y=c.geometria[i].exterior.xy
                ax.plot(x, y, '--',color='red')
    else:
        ax.plot(x,y,'--',color='gray')

for c in crl:
    x,y=df.getCausticPoints(c,pixel_units=True)
    if (c.principale):
        ax.plot(x,y,'-',color='black')
    else:
        ax.plot(x,y,'-',color='orange')
ax.set_xlim([0,2047])
ax.set_ylim([0,2047])
plt.show()



import pickle
PIK = "pickle_test.dat"

zsarr=np.logspace(np.log10(0.7),np.log10(7.0),10)
cs_list=[]
for zs_ in zsarr:
    df.change_redshift(zs_)

    detA = (1.0 - df.ka) ** 2 - (df.g1 ** 2 + df.g2 ** 2)
    y1 = df.theta1 - df.a1 + 2 * df.grid_pixel
    y2 = df.theta2 - df.a2 + 2 * df.grid_pixel
    mus = pymupds.mupds_triangle(y1 - df.theta.min(), y2 - df.theta2.min(), df.grid_pixel, detA, nray=len(theta))
    fig1,ax1=plt.subplots(1,1,figsize=(10,10))
    ax1.imshow(np.log10(mus.T),vmax=1.7,origin='low')
    plt.savefig(str(zs_)+'.png')

    tl = df.tancl()
    ctl = df.getCaustics(tl)
    hull, points, harea = df.causticRegion(caut=ctl)
    for i in range(len(tl)):
        tl[i].toArcsec(df.nray,df.grid_pixel)
        ctl[i].toArcsec(df.nray,df.grid_pixel)
    sx,sy,fovsp=df.fovSP()
    cs={'zs': zs_,
        'MI': df.multImaCrossSection(),
        'GGSL': df.ggslCrossSection(minsize=0.5,maxsize=5.0,dmax=200.0),
        'CAUAREA': harea,
        'FOVSP': fovsp,
        'CritLines': tl,
        'CauLines': ctl
        }
    print (cs)
    cs_list.append(cs)
plt.show()

import pandas as pd
dff=pd.DataFrame(cs_list)
with open(PIK, "wb") as f:
    pickle.dump(dff, f)

fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.plot(dff.zs,dff.GGSL,'o-')
ax.plot(dff.zs,dff.MI,'o-')
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()


#print (df.multImaCrossSection())
#print (df.ggslCrossSection(minsize=0.9,maxsize=3.0,dmax=100.0))
#df.change_redshift(5.41)
#print (df.multImaCrossSection())
#print (df.ggslCrossSection(minsize=0.9,maxsize=3.0,dmax=100.0))

fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(df.ka,vmax=3.0,origin='low',extent=[-100,100,-100,100])
tl=df.tancl()
rl=df.radcl()



print (len(tl))
ctl=df.getCaustics(tl)
crl=df.getCaustics(rl)
for c in tl:
    x,y=df.getCritPoints(c,pixel_units=False)
    if (c.principale):
        ax.plot(x,y,'-',color='white')
    else:
        ax.plot(x,y,'-',color='yellow')
for c in rl:
    x,y=df.getCritPoints(c,pixel_units=False)
    ax.plot(x,y,'-',color='white')

for c in ctl:
    x,y=df.getCausticPoints(c,pixel_units=False)
    if (c.principale):
        ax.plot(x,y,'-',color='white')
    else:
        ax.plot(x,y,'-',color='orange')

hull,points,harea=df.causticRegion(caut=ctl)

for simplex in hull.simplices:
    ax.plot(points[simplex, 0], points[simplex, 1], 'k--',color='white')

sx,sy,sa=df.fovSP()
ax.plot(sx,sy,'--',color='white')

plt.show()

tot=0
for i in range(1,len(tl)):
    tot+=ctl[i].area*df.pixel_scale**2

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(1.0/(1.-df.ka-np.sqrt(df.g1**2+df.g2**2)),extent=[-100,100,-100,100],origin='low',vmin=-10,vmax=10)
for c in tl:
    x, y = df.getCritPoints(c, pixel_units=False)
    plt.plot(x, y, ':',color='white')
fig.savefig('testdeta.png',dpi=600)


y1,y2,x1,x2=df.multImaRegion()
fig,ax=plt.subplots(1,2,figsize=(20,10))
ax[0].plot(y1,y2,',')
for c in ctl:
    x,y=df.getCausticPoints(c,pixel_units=False)
    if (c.principale):
        ax[0].plot(x,y,'-',color='black')
    else:
        ax[0].plot(x,y,'-',color='red')
for c in crl:
    x,y=df.getCausticPoints(c,pixel_units=False)
    ax[0].plot(x,y,'-',color='black')

ax[1].plot(x1,x2,',')
for c in tl:
    x,y=df.getCritPoints(c,pixel_units=False)
    if (c.principale):
        ax[1].plot(x,y,'-',color='black')
    else:
        ax[1].plot(x,y,'-',color='red')
for c in rl:
    x,y=df.getCritPoints(c,pixel_units=False)
    ax[1].plot(x,y,'-',color='black')

x1_,x2_=1.11412,-12.6071
a1_,a2_=df.getAngle(1.11412,-12.6071)
y1_,y2_=x1_-a1_,x2_-a2_

ax[0].plot(y1_,y2_,'+',color='red',lw=3)
ax[1].plot(x1_,x2_,'+',color='red',lw=3)

for i in range(2):
    ax[i].set_xlim([-100,100])
    ax[i].set_ylim([-100,100])
fig.savefig('testmapping.png',dpi=300)
plt.show()
