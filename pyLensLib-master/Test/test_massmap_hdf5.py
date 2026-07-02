from pyLensLib.cluster import cluster
from pyLensLib.raytracer import raytracer
from pyLensLib.deflector import deflector
import numpy as np
import pandas as pd
import os

def getCenter(clnum,snap):
    for root, dirs, files in os.walk('/Users/massimo/stiva/300/GIZMO-SIMBA/NewMDCLUSTER_'+clnum+'/'):
        for file in files:
            if file.startswith('GIZMO-NewMDCLUSTER_'+clnum+'.snap_'+snap) & file.endswith('AHF_halos'):
                centers_file = os.path.join(root, file)
    #centers_file = '/data4/niftydata/TheThreeHundred/data/catalogues/AHF/GadgetX/NewMDCLUSTER_'+clnum+'/GadgetX-NewMDCLUSTER_'+clnum+'.snap_'+snap_num+'.AHF_halos'
    df = pd.read_csv(centers_file,header=None,skiprows=1,delim_whitespace=True)
    xc=df.iloc[ : , 5 ].values[0]/1000.0
    yc=df.iloc[ : , 6 ].values[0]/1000.0
    zc=df.iloc[ : , 7 ].values[0]/1000.0
    mvir = df.iloc[ :, 3].values[0]
    rvir = df.iloc[ :, 11].values[0]
    cnfw = df.iloc[ :, 42].values[0]
    ba = df.iloc[ :, 24].values[0]
    ca = df.iloc[ :, 25].values[0]
    eax = df.iloc[ :, 26].values[0]
    eay = df.iloc[ :, 27].values[0]
    eaz = df.iloc[ :, 28].values[0]
    ebx = df.iloc[ :, 29].values[0]
    eby = df.iloc[ :, 30].values[0]
    ebz = df.iloc[ :, 31].values[0]
    ecx = df.iloc[ :, 32].values[0]
    ecy = df.iloc[ :, 33].values[0]
    ecz = df.iloc[ :, 34].values[0]
    InTens = [[eax,eay,eaz],[ebx,eby,ebz],[ecx,ecy,ecz]]
    data = {'xc': xc, 'yc': yc, 'zc': zc, 'mvir': mvir, 'rvir': rvir, 'cnfw': cnfw, 'ba': ba, 'ca': ca, 'InTens': InTens}
    return xc,yc,zc,data

snap='110'
clnum = '0001'
cluster_file = "/Users/massimo/stiva/300/GIZMO-SIMBA/snap_110.hdf5"

xc,yc,zc,data = getCenter(clnum,snap)
cl = cluster(snapshot=cluster_file,parttype=[0],xc=xc,yc=yc,zc=zc,hdf5=True)




print (xc,yc,zc)
print (cl.zl,cl.omega,cl.lambd,cl.h)



parttype = [0,1,4,5]

npix=2048
img = np.zeros((npix,npix))
fov=200.0
fov_mpc=fov * cl.co.angular_diameter_distance(cl.zl).value*np.pi/180./3600.

for itype in parttype:
    cl=cluster(snapshot=cluster_file,parttype=[itype],xc=xc,yc=yc,zc=zc,IT=np.array([[1,0,0],[0,0,1],[0,1,0]]),alignIT=True,hdf5=True)
    this_img = cl.massMapSPH(nb=20,pX=90.,pY=90,npix=npix,xmin=-fov_mpc/2.0,xmax=fov_mpc/2.0,ymin=-fov_mpc/2.0,ymax=fov_mpc/2.0,zmin=-10.0,zmax=10.0,min_hsml=0.0001,max_hsml=0.1)
    img = img.copy() + this_img




kwargs = {'zl': cl.zl, 'zs': 3.0, 'fov': fov}
rayt=raytracer(cl.co,img,Nray=2048,FOVray=fov,fromfile=False,**kwargs)

kwargs_def={'zl': cl.zl, 'zs': kwargs['zs']}
df = deflector(cl.co,angx=rayt.a1,angy=rayt.a2,**kwargs_def)
theta=np.linspace(-fov/2.,fov/2.,2048)
df.setGrid(theta=theta)

print ('mass %10.4e' % img.sum())
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(np.sqrt(img),origin='lower',extent=[-fov/2,fov/2,-fov/2,fov/2])

tancl = df.tancl()
for cl in tancl:
    x,y=df.getCritPoints(cl)
    ax.plot(x,y,'-',color='white')

radcl = df.radcl()
for cl in radcl:
    x,y=df.getCritPoints(cl)
    ax.plot(x,y,'-',color='white')


plt.show()

