import pandas as pd

import pyLensLib.lenstool as lst
import astropy.io.fits as fits
import os
from scipy.spatial import ConvexHull
import shapely
import shapely.geometry
from shapely.ops import polygonize, unary_union
import pandas as pd


class myggsl:

    def __init__(self,cluster='A370',zs=1.0, dmax = 80):
        self.cluster = cluster
        self.zs = zs
        self.lens_init()
        if cluster == 'A2390':
            self.x1, self.x2 = self.getDelimitingPoints(dmax=dmax,mtype='gravity')
        else:
            self.x1, self.x2 = self.getDelimitingPoints(dmax=dmax,mtype='lenstool')
        return


    def lens_init(self):
        # path to the deflection angle maps: RECOMMENDATION: should be stored in a configuration file
        self.path_to_angles = {
            'S1063': '/Users/maxmen3/stiva/pietro_models/',
            'M0416_B22': '/Users/maxmen3/stiva/pietro_models/',
            'M0416_canucs': '/Users/maxmen3/stiva/canucs_models/',
            'M1206pl': '/Users/maxmen3/stiva/pietro_models/',
            'PSZ1G311_200': '/Users/maxmen3/stiva/clusters/PSZ1G311/delens/',
            'A370': '/Users/maxmen3/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
            'A2744': '/Users/maxmen3/stiva/pietro_models/',
            'M0717': '/Users/maxmen3/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
            'M1149': '/Users/maxmen3/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
            'M0329': '/Users/maxmen3/stiva/gabriel_models/best_fit/',
            'M1931': '/Users/maxmen3/stiva/gabriel_models/best_fit/',
            'M2129': '/Users/maxmen3/stiva/gabriel_models/best_fit/',
            'R2129': '/Users/maxmen3/stiva/gabriel_models/best_fit/',
            'PLCK-G287': '/Users/maxmen3/stiva/daddona_models/PLCK-G287/',
            'elgordo': '/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/',
            'A2390': '/Users/maxmen3/stiva/abriola_models/Deflection_Maps_A2390_Gravity/'
        }

        self.file_prefix = self.path_to_angles[self.cluster] + self.cluster
        #print (self.file_prefix)
        # get the lens redshift from the lenstool parameter file and create the deflector
        # RECOMMENDATION: we should  get these info in a different way if the GUI has to be used with
        # lens models (e.g., from gravity.jl)
        zl = float(lst.getLensRedshift(self.file_prefix + '.par'))
        self.zl = zl

        try:
            self.lens = lst.create_deflector(
                parfile=self.file_prefix + '.par',
                filex=self.file_prefix + '_angx.fits',
                filey=self.file_prefix + '_angy.fits',
                filepot=self.file_prefix + '_pot.fits',
                usePotential=False,
                zl=zl, zs=self.zs, zsnorm=1.0, resc_fact=1.0, compute_potential=False)
        except Exception as e:
            print(f"Error creating deflector: {e}")
            raise

        # if the file exists, load rgb.fits image of the cluster
        if os.path.exists(self.file_prefix + '_rgb.fits'):
            with fits.open(self.file_prefix + '_rgb.fits') as hudl:
                rgb_fits_data = hudl[0].data
            rgb_image = np.transpose(rgb_fits_data, (1, 2, 0))
            # rgb_image = np.flip(rgb_image, axis=0)
            rgb_image = rgb_image / np.max(rgb_image)
            self.rgb_image = rgb_image
        else:
            self.rgb_image = self.lens.ka

        # get the field of view from the lenstool parameter file
        # RECOMMENDATION: see above
        lims = lst.getFoV(self.file_prefix + '.par')
        fov = lims[1] - lims[0]
        #print('FOV:', fov, lims)
        # set the grid for the deflector

        # set limits in the source and lens planes. These limits will be adjusted
        # when the user zooms in the plots
        self.beta1_lim = [-fov / 2., fov / 2.]  # [lims[0], lims[1]]
        self.beta2_lim = [-fov / 2., fov / 2.]  # [lims[2], lims[3]]
        self.theta1_lim = self.beta1_lim
        self.theta2_lim = self.beta2_lim

        # set the extent of the plots. These values will be used to reset the zoom
        # and to freeze the size of displayed images in imshow calls

        # self.extent = [self.beta1_lim[0], self.beta1_lim[1], self.beta2_lim[0], self.beta2_lim[1]]
        self.extent = [-fov / 2., fov / 2., -fov / 2., fov / 2.]
        self.extent1 = self.theta1_lim.copy()
        self.extent2 = self.theta2_lim.copy()

    def getDelimitingPoints(self,dmax=80,mtype='lenstool'):
        if mtype == 'lenstool':
            clmemb_ = lst.getClMembers(self.file_prefix + '.par')
            lims = lst.getFoV(self.file_prefix + '.par')
            xcen = 0.5 * (lims[1] + lims[0])
            ycen = 0.5 * (lims[3] + lims[2])

            clmemb = clmemb_.loc[(np.abs(clmemb_.x_centre.values - xcen) < dmax) &
                                 (np.abs(clmemb_.y_centre.values - ycen) < dmax)]

            xp = clmemb.x_centre.values - xcen
            yp = clmemb.y_centre.values - ycen

        elif mtype == 'gravity':
            dfclmemb = pd.read_csv(self.file_prefix+'_clmemb.csv',header=0,sep=',')
            ra = dfclmemb.RA_deg.values
            dec = dfclmemb.Dec_deg.values
            ra_ref,dec_ref = lst.getRef_RA_DEC(self.file_prefix+'.par')

            # compute positions in arcsec relative to ra_ref and dec_ref
            x = (ra-ra_ref)*3600.0*np.cos(np.deg2rad(dec_ref))
            y = (dec-dec_ref)*3600.0

            ind = (np.abs(x) < dmax) & (np.abs(y) < dmax)
            xp = x[ind]
            yp = y[ind]

        # compute the FOVsp
        points = np.column_stack((xp, yp))
        hull = ConvexHull(points)
        x1 = list(xp[hull.vertices])
        x2 = list(yp[hull.vertices])
        x1.append(x1[0])
        x2.append(x2[0])
        return x1, x2


    def computeGGSLprob(self,minsize=0.1,maxsize=3.0):
        x1 = np.array(self.x1) / self.lens.pixel_scale + self.lens.nray1 / 2.
        x2 = np.array(self.x2) / self.lens.pixel_scale + self.lens.nray1 / 2.
        y1, y2 = self.lens.mapCrit2Cau(x1, x2)
        y1 = (y1 - self.lens.nray1 / 2.) * self.lens.pixel_scale
        y2 = (y2 - self.lens.nray1 / 2.) * self.lens.pixel_scale

        vs = zip(y1, y2)
        ls = shapely.geometry.LineString(vs)
        mls = unary_union(ls)
        mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
        ggsl = self.lens.ggslCrossSection(minsize=minsize, maxsize=maxsize) / mp.area  # maxsize was 5
        return ggsl

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax = plt.subplots(1,2, figsize = (18,10), sharey=True)
    cluster_list = ['A2390', 'S1063','A2744', 'M0416_B22', 'M1206pl', 'PSZ1G311_200', 'PLCK-G287', 'elgordo']
    cluster_labels = ['A2390', 'AS1063', 'A2744', 'M0416', 'M1206', 'PSZ1G311', 'PLCK-G287', 'El Gordo']
    zs = np.linspace(1.5,6.0,10)
    ggsl_total = []
    zl_=[]
    ggsl_z6=[]

    for cluster in cluster_list:
        print('Working with cluster %s' % (cluster))
        model = myggsl(cluster=cluster)
        ggsl = []
        zl_.append(model.zl)

        for zs_ in zs:
            # compute the ggsl cross section
            model.lens.change_redshift(zs_)
            #ggsl_ = model.lens.ggslCrossSection(minsize=0.5, maxsize=3.0, dmax=200.0)
            #areasp_ = model.lens.fovSP()
            #print (ggsl_,areasp_)
            ggsl_ = model.computeGGSLprob(minsize=0.5,maxsize=3.0)
            ggsl.append(ggsl_*1e6)
        #if (cluster == 'A2390'):
        #    ax[0].plot(zs,ggsl,'-',label=cluster,lw=3)
        ggsl_total.append(ggsl)
        ggsl_z6.append(ggsl_*1e6)

    # compute the median and 25% and 75% quartiles of ggsl as a function of zs
    ggsl_total = np.array(ggsl_total)
    ggsl_median = np.median(ggsl_total, axis=0)
    ggsl_25 = np.percentile(ggsl_total, 25, axis=0)
    ggsl_75 = np.percentile(ggsl_total, 75, axis=0)
    ax[0].plot(zs,ggsl_median,'k-',label='observations',lw=3)
    ax[0].fill_between(zs, ggsl_25, ggsl_75, color='gray', alpha=0.5)

    # read simulation data
    df_dianoga_m = pd.read_csv('dianoga_median.csv',header=None)
    df_dianoga_m.columns = ['zs', 'Pggsl']
    df_dianoga_m = df_dianoga_m.sort_values(by='zs')
    df_dianoga_l = pd.read_csv('dianoga_lower.csv',header=None)
    df_dianoga_l.columns = ['zs', 'Pggsl']
    df_dianoga_l = df_dianoga_l.sort_values(by='zs')
    df_dianoga_u = pd.read_csv('dianoga_upper.csv',header=None)
    df_dianoga_u.columns = ['zs', 'Pggsl']
    df_dianoga_u = df_dianoga_u.sort_values(by='zs')
    ax[0].plot(df_dianoga_m.zs, df_dianoga_m.Pggsl, '-',color='violet', label='Dianoga (M20)', lw=3)
    ax[0].fill_between(df_dianoga_m.zs, df_dianoga_l.Pggsl, df_dianoga_u.Pggsl, color='violet', alpha=0.3)

    df_B20 = pd.read_csv('B20.csv',header=None)
    df_B20.columns = ['zs', 'Pggsl']
    df_B20 = df_B20.sort_values(by='zs')

    ax[0].plot(df_B20.zs, df_B20.Pggsl, '--', color='magenta', label='10xB20 (M22)', lw=3)

    df_R15 = pd.read_csv('R15.csv',header=None)
    df_R15.columns = ['zs', 'Pggsl']
    df_R15 = df_R15.sort_values(by='zs')

    ax[0].plot(df_R15.zs, df_R15.Pggsl, '--', color='navy', label='R15 (M22)', lw=3)

    df_RF18 = pd.read_csv('RF18.csv',header=None)
    df_RF18.columns = ['zs', 'Pggsl']
    df_RF18 = df_RF18.sort_values(by='zs')

    ax[0].plot(df_RF18.zs, df_RF18.Pggsl, '--', color='cyan', label='RF18 (M22)', lw=3)

    ax[0].set_xlabel(r'$z_s$',fontsize=20)
    ax[0].set_ylabel(r'$P_{ggsl}(z_s)$ [$10^{-6}$]',fontsize=20)
    ax[0].set_yscale('log')
    ax[0].set_ylim(10,30000)
    ax[0].set_xlim(1.5,6.0)
    ax[0].legend(fontsize=20,loc='upper left')

    from adjustText import adjust_text
    #ax[1].plot(zl_,ggsl_z6,'o',label='z=6',lw=3)
    # add a vertical label near each point indicating the cluster
    for i, cluster in enumerate(cluster_list):
        ax[1].plot(zl_[i], ggsl_z6[i],'o',color='gray')
    #    ax[1].text(zl_[i], ggsl_z6[i]+10, cluster, fontsize=20, ha='center', va='bottom')
    texts = [
        ax[1].text(z, y, label, fontsize=20, ha='center', va='bottom')
        for z, y, label in zip(zl_, ggsl_z6, cluster_labels)
    ]
    # Automatically adjust labels to reduce overlap (works with log y)
    adjust_text(
        texts,
        x=zl_,
        y=ggsl_z6,
        ax=ax[1],
        only_move={'points': 'y', 'texts': 'y'},  # keep x fixed, move along y
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4),
        force_points=(0.2, 0.2),
        force_text=(0.3, 0.3),
        #arrowprops=dict(arrowstyle='-', color='0.3', lw=0.8)  # draw leader lines if moved
    )

    ax[1].set_yscale('log')
    ax[1].set_ylim(10,30000)
    ax[1].set_xlim(0.05,1.0)
    ax[1].set_xlabel(r'$z_l$', fontsize=20)

    # read simulation data:
    df_gadgetx_m = pd.read_csv('GADGET-X_median.csv',header=None)
    df_gadgetx_m.columns=['zl','Pggsl']
    # sort by zl
    df_gadgetx_m = df_gadgetx_m.sort_values(by='zl')
    df_gadgetx_l = pd.read_csv('GADGET-X_lower.csv',header=None)
    df_gadgetx_l.columns = ['zl', 'Pggsl']
    df_gadgetx_l = df_gadgetx_l.sort_values(by='zl')
    df_gadget_u = pd.read_csv('GADGET-X_upper.csv',header=None)
    df_gadget_u.columns = ['zl', 'Pggsl']
    df_gadget_u = df_gadget_u.sort_values(by='zl')

    # same for GIZMO-SIMBA
    df_gizmosimba_m = pd.read_csv('GIZMO-SIMBA_median.csv',header=None)
    df_gizmosimba_m.columns = ['zl', 'Pggsl']
    df_gizmosimba_m = df_gizmosimba_m.sort_values(by='zl')
    df_gizmosimba_l = pd.read_csv('GIZMO-SIMBA_lower.csv',header=None)
    df_gizmosimba_l.columns = ['zl', 'Pggsl']
    df_gizmosimba_l = df_gizmosimba_l.sort_values(by='zl')
    df_gizmosimba_u = pd.read_csv('GIZMO-SIMBA_upper.csv',header=None)
    df_gizmosimba_u.columns = ['zl', 'Pggsl']
    df_gizmosimba_u = df_gizmosimba_u.sort_values(by='zl')


    # plot median and shaded area between lower and upper
    ax[1].plot(df_gadgetx_m.zl, df_gadgetx_m.Pggsl, 'r-', label='GADGET-X (M23)', lw=3)
    ax[1].fill_between(df_gadgetx_m.zl, df_gadgetx_l.Pggsl, df_gadget_u.Pggsl, color='red', alpha=0.3)

    ax[1].plot(df_gizmosimba_m.zl, df_gizmosimba_m.Pggsl, 'b-', label='GIZMO-SIMBA (M23)', lw=3)
    ax[1].fill_between(df_gizmosimba_m.zl, df_gizmosimba_l.Pggsl, df_gizmosimba_u.Pggsl, color='blue', alpha=0.3)
    ax[1].legend(loc='lower right',fontsize=20)


    # set the ticklabels as fontsize 20
    for i in range(2):
        ax[i].tick_params(axis='both', which='major', labelsize=20)
    plt.subplots_adjust(wspace=0)

    #fig.tight_layout()
    fig.savefig('summary_comp.pdf',dpi=300)

    plt.show()


