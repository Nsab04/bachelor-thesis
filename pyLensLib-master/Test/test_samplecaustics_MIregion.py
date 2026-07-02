from pyLensLib.piemd import piemd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import geopandas as gpd
from icecream import ic

co = FlatLambdaCDM(Om0=0.32, H0=67.0)

kwargs1 = {'zl': 0.5, 'zs': 2.0, 'sigma0': 200.0, 'q': 0.5, 'pa': 0.0,
           'theta_c': 0.0, 'theta_t': 5.0, 'x1': 0.0, 'x2': 0.0}

kwargs_sub = {'zl': 0.5, 'zs': 2.0, 'sigma0': 80.0, 'q': 0.8, 'pa': np.pi/4,
           'theta_c': 0.0, 'theta_t': 2.0, 'x1': 0.9, 'x2': 0.9}

p1 = piemd(co, **kwargs1)
psub = piemd(co, **kwargs_sub)
theta = np.linspace(-5, 5, 512)
p1.setGrid(theta=theta)
psub.setGrid(theta=theta)
p1.combinewith(psub)

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# plot velocity curve
velocity_curve = False
if velocity_curve:
    r = np.logspace(-4, 0, 1000)
    vc = p1.vcirc(r)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(r, vc, '-')
    # ax.set_xscale('log')
    plt.show()

caustic_plot = True
if caustic_plot:
    from shapely.geometry import Polygon

    tcl = p1.tancl()
    tca = p1.getCaustics(tcl)
    buffer_size = 4

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for t in tca:
        x, y = p1.getCausticPoints(t)
        ic (len(x))
        ax.plot(x, y, '-', color='red', linewidth=3, zorder=10)

        ic(t.geometria)
        t.addBuffer(buffer_size=buffer_size)
        ic(t.buffer)
        px, py = p1.random_sources_in_caustic(t,buffer_size=buffer_size,number=10000)
        ic('Points inside the caustic:', px.size)

        ax.plot(px,py,'o',color='green',markersize=2)

        px, py = p1.random_sources_along_caustic(t, buffer_size=buffer_size, number=10000)
        ic('Points along the caustic:', px.size)

        ax.plot(px, py, 'o', color='orange', markersize=2)

        x, y = p1.getBufferCausticPoints(t)
        ax.plot(x, y, '-', color='blue')

    #ic(p1.multImaCrossSection(),p1.multImaCrossSection(buffer_size=buffer_size))



    for t in tcl:
        x, y = p1.getCritPoints(t)
        ax.plot(x, y, '-', color='red', linewidth=3)

    ax.plot([0,0],[-1,1],':',color='black')
    ax.plot([-1, 1], [0, 0], ':',color='black')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    #plt.show()

    tcl = p1.tancl()
    rcl = p1.radcl()
    tca = p1.getCaustics(tcl)
    rca = p1.getCaustics(rcl)

    ic('Computing region of image multiplicity...')
    cc = p1.causticsUnaryUnion(buffer_size=buffer_size)
    xp, yp = p1.multImaRegion(buffer_size=buffer_size)
    id('...done')

    boundary = gpd.GeoSeries(cc)
    ic(boundary.area.values*p1.pixel_scale**2,p1.multImaCrossSection(buffer_size=buffer_size))
    boundary.plot(color='yellow')
    for t in rca:
        x, y = p1.getCausticPoints(t,pixel_units=True)
        plt.plot(x, y, '-', color='blue', linewidth=3)
    for t in tca:
        x, y = p1.getCausticPoints(t,pixel_units=True)
        plt.plot(x, y, '-', color='red', linewidth=3)

    plt.show()

    multi_ima_map = np.zeros((p1.nray1,p1.nray2))

    ix = ((xp - p1.thetax[0]) / p1.pixel_scale).astype(int)
    iy = ((yp - p1.thetay[0]) / p1.pixel_scale).astype(int)
    multi_ima_map[ix,iy] = 1.0

    fig,ax=plt.subplots(1,1,figsize=(20,20))
    ax.imshow(multi_ima_map,origin='lower',
              extent=[p1.thetax[0],p1.thetax[-1],p1.thetay[0],p1.thetay[-1]])

    for t in tcl:
        x, y = p1.getCritPoints(t)
        ax.plot(x, y, '-', color='green', linewidth=3,zorder=10)

    for t in rcl:
        x, y = p1.getCritPoints(t)
        ax.plot(x, y, '-', color='green', linewidth=3,zorder=10)
    #for c_ in cc:
    #    xx, yy = c_.exterior.coords.xy
    #    plt.plot(xx,yy,'-')
    plt.show()

    # test difference between regions:
    cc_ext = p1.causticsUnaryUnion(buffer_size=buffer_size)
    cc_int = p1.causticsUnaryUnion(buffer_size=-buffer_size)

    cc_ = cc_ext.difference(cc_int)
    boundary_ = gpd.GeoSeries(cc_)
    boundary_.plot(color='yellow')
    ic('Boundary area:',boundary_.area)

    for t in rca:
        x, y = p1.getCausticPoints(t,pixel_units=True)
        plt.plot(x, y, '-', color='blue', linewidth=3)
    for t in tca:
        x, y = p1.getCausticPoints(t,pixel_units=True)
        plt.plot(x, y, '-', color='red', linewidth=3)
    plt.show()

