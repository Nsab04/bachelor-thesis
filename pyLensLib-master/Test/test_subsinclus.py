from pyLensLib.cluster import cluster
from pyLensLib.gadget import *
from pyLensLib.raytracer import raytracer
from pyLensLib.subfind import subfind
from pyLensLib.deflector import deflector
from pyLensLib.observation import observation
from pyLensLib.sersic import sersic
import numpy as np
import astropy.io.fits as fits

import matplotlib

matplotlib.use('TkAgg')

def getBlock(subfind_file,blockname,numpydt):
    head = snapshot_header(subfind_file)
    format = head.format
    swap = head.swap
    npart = head.npart
    dt = np.dtype(numpydt)
    offset, blocksize = find_block(subfind_file, format, swap, blockname, 54)
    curpartnum = np.int32(0)
    cur_species_offset = np.zeros(6, np.int64)
    parttype = -1
    for j in range(6):
        cur_species_offset[j] = curpartnum
        curpartnum += npart[j]

    if parttype >= 0:
        actual_curpartnum = npart[parttype]
        add_offset = cur_species_offset[parttype]
    else:
        actual_curpartnum = curpartnum
        add_offset = np.int32(0)

    f = open(subfind_file, 'rb')
    f.seek(offset, os.SEEK_CUR)

    #f.seek(offset, os.SEEK_CUR)
    #curdat = np.fromfile(f, dtype=dt, count=actual_curpartnum)  # read data
    curdat = np.fromfile(f, dtype=dt, count=int(blocksize/np.dtype(dt).itemsize))
    f.close()
    if swap:
        curdat.byteswap(True)

    return curdat

#snapshot = '/Users/massimo/projects/ggsl/testing_codes/massmaps_python/BH_2015/snap_058'
#snapshot = '/Users/massimo/stiva/Dianoga/new_elena/BH_2015_snaps/snap_065'
snapshot = '/Users/massimo/stiva/300/testCodes/snap_065'
#subfind_file = "/Users/massimo/stiva/Dianoga/new_elena/Subfind_AGN/D1/sub_058.0"
subfind_file = "/Users/massimo/stiva/Dianoga/new_elena/Subfind_AGN/D1/sub_065.0"
sf=subfind(subfind_file)
xc,yc,zc=sf.gpos[0]
cl = cluster(snapshot=snapshot,parttype=[0,1,4,5],xc=xc,yc=yc,zc=zc)

#print (len(cl.mass))

#import matplotlib.pyplot as plt

#list_format2_blocks(subfind_file)
#read_block(subfind_file, "PID", verbose=1,parttype=2)

pid = getBlock(subfind_file,'PID ',np.uint32)
goff = getBlock(subfind_file,'GOFF',np.uint32)
soff = getBlock(subfind_file,'SOFF',np.uint32)


"""
head = snapshot_header(subfind_file)
format = head.format
swap = head.swap
npart=head.npart
dt = np.dtype(np.uint32)
offset, blocksize = find_block(subfind_file, format, swap, "PID ", 54)


curpartnum = np.int32(0)
cur_species_offset = np.zeros(6, np.int64)
parttype=2
for j in range(6):
    cur_species_offset[j] = curpartnum
    curpartnum += npart[j]

if parttype >= 0:
    actual_curpartnum = npart[parttype]
    add_offset = cur_species_offset[parttype]
else:
    actual_curpartnum = curpartnum
    add_offset = np.int32(0)

f = open(subfind_file, 'rb')
f.seek(offset + add_offset * np.dtype(dt).itemsize, os.SEEK_CUR)
curdat = np.fromfile(f, dtype=dt, count=actual_curpartnum)  # read data
f.close()
if swap:
    curdat.byteswap(True)


offset, blocksize = find_block(subfind_file, format, swap, "GOFF", 54)
print (offset, blocksize)


"""

grp=0


print ('number of subhalos in group %d: %d' % (grp,sf.nsub[grp]))
print ('index of first subhalo in group %d: %d' % (grp,sf.fsub[grp]))
print ('number of particles in the first subhalo of group %d: %d' % (grp,sf.slen[sf.fsub[grp]]))
isel= np.nonzero(np.in1d(cl.pid,pid[soff[sf.fsub[grp]]:soff[sf.fsub[grp]+1]]))[0]
print (isel.shape,soff[sf.fsub[grp]],soff[sf.fsub[grp]+1],
       soff[sf.fsub[grp]+1]-soff[sf.fsub[grp]])

"""
rmax=1.0
map=cl.massMap(pX=0.0,pY=0.0, npix=512,xmin=-rmax,xmax=rmax,ymin=-rmax,ymax=rmax,zmin=-rmax,zmax=rmax,sigma=1)
fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.imshow(map,origin='low',extent=[-rmax,rmax,-rmax,rmax],alpha=0.8,cmap='gray_r')
sid=np.zeros_like(cl.pid)
spos=sf.spos/cl.h/(1.0+cl.zl)

spos[:,0]=spos[:,0]-cl.xc
spos[:,1]=spos[:,1]-cl.yc
spos[:,2]=spos[:,2]-cl.zc
d=np.sqrt((spos**2).sum(axis=1))

for i in range(1,sf.nsub[grp]):
    print(i)
    if d[i] < rmax:
        i0=sf.fsub[grp]
        isel= np.nonzero(np.in1d(cl.pid,pid[soff[i0+i]:soff[i0+i+1]]))[0]
        sid[isel]=i
        ax.plot(cl.pos[isel,0]-cl.xc,cl.pos[isel,1]-cl.yc,',',zorder=0)
plt.show()
exit()
"""



from mpl_toolkits import mplot3d

"""
for i in range(1,sf.nsub[grp]):
    i0=sf.fsub[grp]
    isel= np.nonzero(np.in1d(cl.pid,pid[soff[i0+i]:soff[i0+i+1]]))[0]
#    print (isel.sum())
    plt.plot(cl.pos[isel,0],cl.pos[isel,1],',')

plt.plot(xc/(1.0+cl.zl)/cl.h,yc/(1.0+cl.zl)/cl.h,'o',color='red')
plt.show()
"""

from vispy import app, scene, visuals


Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
canvas = scene.SceneCanvas(keys='interactive', show=True,bgcolor='white')
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 10
view.camera.set_range(x=(-5,5),y=(-5,5),z=(-5,5))

from matplotlib.pyplot import cm
color=iter(cm.rainbow(np.linspace(0,1,sf.nsub[grp])))

#colors=['red','blue','green','cyan','orange','yellow','magenta','pink','green','blue']
"""
for i in range(1,sf.nsub[grp]):
    c = next(color)
    print (i)
    i0=sf.fsub[grp]
    isel= np.nonzero(np.in1d(cl.pid,pid[soff[i0+i]:soff[i0+i+1]]))[0]
    pos = np.stack((cl.pos[isel,0]-cl.xc,cl.pos[isel,1]-cl.yc,cl.pos[isel,2]-cl.zc),axis=1)
    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.set_data(pos, face_color=c, symbol='o', size=2,
                edge_width=0.05, edge_color='blue')
    view.add(p1)
#axis = scene.visuals.XYZAxis(parent=view.scene)
"""
i0=sf.fsub[grp]
nsub=sf.nsub[grp]
isel= np.nonzero(np.in1d(cl.pid,pid[soff[i0]:soff[i0+1]]))[0]
pos = np.stack((cl.pos[isel,0]-cl.xc,cl.pos[isel,1]-cl.yc,cl.pos[isel,2]-cl.zc),axis=1)
#pos = np.stack((cl.pos[:,0]-cl.xc,cl.pos[:,1]-cl.yc,cl.pos[:,2]-cl.zc),axis=1)
p1 = Scatter3D(parent=view.scene)
p1.set_gl_state('translucent', blend=True, depth_test=True)
p1.set_data(pos, face_color='red', symbol='o', size=0.5,
                edge_width=0.05, edge_color='blue')
view.add(p1)
isel= np.nonzero(np.in1d(cl.pid,pid[soff[i0+1]:soff[i0+nsub]]))[0]
pos = np.stack((cl.pos[isel,0]-cl.xc,cl.pos[isel,1]-cl.yc,cl.pos[isel,2]-cl.zc),axis=1)
p1 = Scatter3D(parent=view.scene)
p1.set_gl_state('translucent', blend=True, depth_test=True)
p1.set_data(pos, face_color='blue', symbol='o', size=3,
                edge_width=0.05, edge_color='blue')
view.add(p1)
maxext=3.0
xax = scene.Axis(pos=[[-maxext, 0], [maxext, 0]], domain=(-maxext,maxext),
                 axis_label='x', axis_label_margin=35, tick_direction=(0, -1),
                 axis_color='k', tick_color='k', text_color='k', font_size=36,
                 parent=view.scene)
yax = scene.Axis(pos=[[0, -maxext], [0, maxext]], domain=(-maxext,maxext),
                 axis_label='y', axis_label_margin=35, tick_direction=(-1, 0),
                 axis_color='k', tick_color='k', text_color='k', font_size=36,
                 parent=view.scene)
zax = scene.Axis(pos=[[maxext, 0], [-maxext, 0]], domain=(-maxext,maxext),
                 axis_label='z', axis_label_margin=35, tick_direction=(0, -1),
                 axis_color='k', tick_color='k', text_color='k', font_size=36,
                 parent=view.scene)
zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

import vispy.io as io
canvas.show()
img=canvas.render()
io.write_png("wonderful.png",img)
app.run()
print ('end!')


"""
print (cl.xc,cl.yc,cl.zc)
fig = plt.figure()
ax3d = plt.axes(projection='3d')
#ax3d.plot3D(cl.pos[:,0]-cl.xc,cl.pos[:,1]-cl.yc,cl.xc,cl.pos[:,1]-cl.zc,',',alpha=0.1)
for i in range(1,sf.nsub[grp]):
    i0=sf.fsub[grp]
    isel= np.nonzero(np.in1d(cl.pid,pid[soff[i0+i]:soff[i0+i+1]]))[0]
#    print (isel.sum())
    ax3d.plot3D(cl.pos[isel,0]-cl.xc,cl.pos[isel,1]-cl.yc,cl.pos[isel,2]-cl.zc,linestyle=None, zdir='z',marker=',',alpha=0.2)
    #ax3d.plot3D(cl.pos[isel, 0], cl.pos[isel, 1], cl.pos[isel, 2], linestyle=None, zdir='z',marker=',',alpha=0.2)
ax3d.set_xlim([-3,3])
ax3d.set_ylim([-3,3])
ax3d.set_zlim([-3,3])
plt.show()
"""

import numpy as np
from scipy import stats
from mayavi import mlab

mu, sigma = 0, 0.1

xyz = pos
kde = stats.gaussian_kde(xyz)
density = kde(xyz)

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot')
pts = mlab.points3d(cl.pos[isel,0]-cl.xc,cl.pos[isel,1]-cl.yc,cl.pos[isel,2]-cl.zc, density, scale_mode='none', scale_factor=0.07)
mlab.axes()
mlab.show()
