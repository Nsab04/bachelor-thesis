import matplotlib.pyplot as plt
from pyLensLib.sersic_numba import sersic
from pyLensLib.pointsrc import pointsrc
import numpy as np
from matplotlib import cm
import matplotlib
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import RectangleSelector
import pyLensLib.lenstool as lst

matplotlib.use('TkAgg')

fontsize = 15

class lens_interact(object):

    def __init__(self, image=None, sersic=False, re=0.2, zs=1.0, showtd=False,
                     with_im_pos=False, **kwargs_plot):


        self.re = re
        self.sersic = sersic
        self.showtd = showtd
        self.with_im_pos = with_im_pos
        self.zs = zs

        if ('beta1_lim' in kwargs_plot):
            self.beta1_lim=kwargs_plot['beta1_lim']
        else:
            self.beta1_lim=[-100.0,100.0]

        if ('beta2_lim' in kwargs_plot):
            self.beta2_lim=kwargs_plot['beta2_lim']
        else:
            self.beta2_lim=[-100.0,100.0]


        if ('theta1_lim' in kwargs_plot):
            self.theta1_lim=kwargs_plot['theta1_lim']
        else:
            self.theta1_lim=[-100.0,100.0]

        if ('theta2_lim' in kwargs_plot):
            self.theta2_lim=kwargs_plot['theta2_lim']
        else:
            self.theta2_lim=[-100.0,100.0]

        if ('cluster' in kwargs_plot):
            self.cluster=kwargs_plot['cluster']
        else:
            self.cluster=['AS1063']

        if ('path' in kwargs_plot):
            self.path=kwargs_plot['path']
        else:
            self.path=['/Users/massimo/stiva/pietro_models/']

        if (len(self.cluster) != len(self.path)):
            print ("ERROR: the cluster labels and paths must have the same size!")
            exit()
        self.cl_id = {self.cluster[i]: i for i in range(len(self.cluster))}
        self.cluster_active = self.cluster[0]

        self.lens_init()

        if image == None:
            # create a plot with two panels only
            self.fig, self.ax = plt.subplots(1, 2, figsize=(20, 10))
            self.ax[0].set_position([0.01,0.5,0.45,0.45])
            self.ax[1].set_position([0.51,0.5,0.45,0.45])

            #plt.subplots_adjust(left=0.25, bottom=0.25)
            fontsize = 15

            self.beta1 = 0.0
            self.beta2 = 0.0

            self.update_td_plot(self.beta1, self.beta2)

            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.mouse_event)

        else:
            self.fig, self.ax = plt.subplots(1, 3, figsize=(20, 10))
            plt.subplots_adjust(left=0.25, bottom=0.25)

            self.beta1 = 0.0
            self.beta2 = 0.0

            self.update_td_plot(self.beta1, self.beta2)

            self.ax[2].imshow(image)

            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.mouse_event)

        for i in range(2):
            self.ax[i].set_aspect('equal')
            self.ax[i].xaxis.set_tick_params(labelsize=fontsize)
            self.ax[i].yaxis.set_tick_params(labelsize=fontsize)

        self.ax[0].set_xlabel(r'$\beta_1$', fontsize=fontsize)
        self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
        self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
        self.ax[1].set_xlabel(r'$\theta_1$', fontsize=fontsize)
        self.ax[1].set_ylabel(r'$\theta_2$', fontsize=fontsize)
        #plt.tight_layout()

        self.axre = self.fig.add_axes([0.1, 0.4, 0.3, 0.03])
        self.re_slider = Slider(
            ax=self.axre,
            label='Effective radius',
            valmin=0.01,
            valmax=3.0,
            valinit=self.re,
        )

        self.axzs = self.fig.add_axes([0.1, 0.35, 0.3, 0.03])
        self.zs_slider = Slider(
            ax=self.axzs,
            label='Source redshift',
            valmin=self.lens.zl,
            valmax=10.0,
            valinit=self.lens.zs,
        )

        self.axreset = self.fig.add_axes([0.8, 0.01, 0.05, 0.03])
        self.bnreset = Button(self.axreset, label='Reset')

        print ('plotting radiobuttons')
        self.rax = self.fig.add_axes([0.7, 0.1, 0.2, 0.3])
        self.radio = RadioButtons(self.rax, self.cluster, active=0)

    def lens_init(self):
        self.file_prefix = self.path[self.cl_id[self.cluster_active]] + \
                       self.cluster[self.cl_id[self.cluster_active]]

        zl = float(lst.getLensRedshift(self.file_prefix + '.par'))

        self.lens = lst.create_deflector(parfile=self.file_prefix + '.par',
                                  filex=self.file_prefix + '_angx.fits',
                                  filey=self.file_prefix + '_angy.fits',
                                  zl=zl, zs=self.zs, zsnorm=1.0, resc_fact=1.0, compute_potential=True)

        lims = lst.getFoV(self.file_prefix + '.par')

        self.beta1_lim = [lims[0], lims[1]]
        self.beta2_lim = [lims[2], lims[3]]
        self.theta1_lim = self.beta1_lim
        self.theta2_lim = self.beta2_lim

        self.beta1_lim_init = self.beta1_lim
        self.beta2_lim_init = self.beta2_lim
        self.theta1_lim_init = self.theta1_lim
        self.theta2_lim_init = self.theta2_lim

        self.tancl = self.lens.tancl()
        self.radcl = self.lens.radcl()

    def update_cluster(self,label):
        self.cluster_active = label
        self.lens_init()
        self.ax[0].clear()
        self.ax[1].clear()
        self.update_td_plot(self.beta1, self.beta2)
        for i in range(2):
            self.ax[i].set_aspect('equal')
            self.ax[i].xaxis.set_tick_params(labelsize=fontsize)
            self.ax[i].yaxis.set_tick_params(labelsize=fontsize)
        self.ax[0].set_xlabel(r'$\beta_1$', fontsize=fontsize)
        self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
        self.ax[1].set_xlabel(r'$\theta_1$', fontsize=fontsize)
        self.ax[1].set_ylabel(r'$\theta_2$', fontsize=fontsize)

    def update_source(self,val):
        self.re = val
        self.ax[0].clear()
        self.ax[1].clear()
        self.update_td_plot(self.beta1, self.beta2)
        for i in range(2):
            self.ax[i].set_aspect('equal')
            self.ax[i].xaxis.set_tick_params(labelsize=fontsize)
            self.ax[i].yaxis.set_tick_params(labelsize=fontsize)
        self.ax[0].set_xlabel(r'$\beta_1$', fontsize=fontsize)
        self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
        self.ax[1].set_xlabel(r'$\theta_1$', fontsize=fontsize)
        self.ax[1].set_ylabel(r'$\theta_2$', fontsize=fontsize)

    def update_defsrc(self,val):
        self.lens.change_redshift(val)
        self.tancl = self.lens.tancl()
        self.radcl = self.lens.radcl()
        self.ax[0].clear()
        self.ax[1].clear()
        self.update_td_plot(self.beta1, self.beta2)
        for i in range(2):
            self.ax[i].set_aspect('equal')
            self.ax[i].xaxis.set_tick_params(labelsize=fontsize)
            self.ax[i].yaxis.set_tick_params(labelsize=fontsize)
        self.ax[0].set_xlabel(r'$\beta_1$', fontsize=fontsize)
        self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
        self.ax[1].set_xlabel(r'$\theta_1$', fontsize=fontsize)
        self.ax[1].set_ylabel(r'$\theta_2$', fontsize=fontsize)

    def reset(self,event):
        self.beta1_lim = self.beta1_lim_init
        self.beta2_lim = self.beta2_lim_init
        self.theta1_lim =  self.theta1_lim_init
        self.theta2_lim = self.theta2_lim_init
        self.ax[0].clear()
        self.ax[1].clear()
        self.update_td_plot(self.beta1, self.beta2)
        for i in range(2):
            self.ax[i].set_aspect('equal')
            self.ax[i].xaxis.set_tick_params(labelsize=fontsize)
            self.ax[i].yaxis.set_tick_params(labelsize=fontsize)
        self.ax[0].set_xlabel(r'$\beta_1$', fontsize=fontsize)
        self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
        self.ax[1].set_xlabel(r'$\theta_1$', fontsize=fontsize)
        self.ax[1].set_ylabel(r'$\theta_2$', fontsize=fontsize)


    def mouse_event(self, event):
        if event.button is MouseButton.LEFT:
            if event.inaxes in [self.ax[0]]:
                self.ax[0].clear()
                self.ax[1].clear()
                self.update_td_plot(event.xdata, event.ydata)
                for i in range(2):
                    self.ax[i].set_aspect('equal')
                    self.ax[i].xaxis.set_tick_params(labelsize=fontsize)
                    self.ax[i].yaxis.set_tick_params(labelsize=fontsize)
                self.ax[0].set_xlabel(r'$\beta_1$', fontsize=fontsize)
                self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
                self.ax[1].set_xlabel(r'$\theta_1$', fontsize=fontsize)
                self.ax[1].set_ylabel(r'$\theta_2$', fontsize=fontsize)
                self.beta1 = event.xdata
                self.beta2 = event.ydata
                print(("clicked on beta1=%10.4f beta2=%10.4f") % (self.beta1, self.beta2))
                #plt.tight_layout()

            if event.inaxes in [self.ax[1]]:
            # calculate beta1, beta2 from deflector

                theta1 = event.xdata
                theta2 = event.ydata

                print (("clicked on theta1=%10.4f theta2=%10.4f") % (theta1, theta2))

                a1,a2 = self.lens.getAngle(theta1,theta2)
                self.beta1, self.beta2 = theta1 - a1, theta2 - a2

                self.ax[0].clear()
                self.ax[1].clear()

                self.update_td_plot(self.beta1, self.beta2)

                for i in range(2):
                    self.ax[i].set_aspect('equal')
                    self.ax[i].xaxis.set_tick_params(labelsize=fontsize)
                    self.ax[i].yaxis.set_tick_params(labelsize=fontsize)
                self.ax[0].set_xlabel(r'$\beta_1$', fontsize=fontsize)
                self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
                self.ax[1].set_xlabel(r'$\theta_1$', fontsize=fontsize)
                self.ax[1].set_ylabel(r'$\theta_2$', fontsize=fontsize)
                #else:
                #    print("Not a clickable region!")
                #    return

            if event.inaxes in [self.axre]:
                self.re_slider.on_changed(self.update_source)

            if event.inaxes in [self.axzs]:
                self.zs_slider.on_changed(self.update_defsrc)

            if event.inaxes in [self.axreset]:
                self.bnreset.on_clicked(self.reset)

            if event.inaxes in [self.rax]:
                self.radio.on_clicked(self.update_cluster)

        if event.button is MouseButton.RIGHT:
            if event.inaxes in [self.ax[0]]:

                props = dict(facecolor='white', alpha=0.0)

                RS = RectangleSelector(self.ax[0],self.select_callback_source,useblit=True,
                                       minspanx=1, minspany=1,
                                       button=[3], spancoords='pixels',
                                       interactive=True,props=props)

                #self.fig.canvas.mpl_connect('key_press_event', RS)
                def toggle_selector(event):

                    if event.key == 't':

                        if RS.active:
                            RS.set_active(False)
                        else:
                            RS.set_active(True)

                self.fig.canvas.mpl_connect('key_press_event', RS)
                plt.gcf().canvas.draw_idle()

            if event.inaxes in [self.ax[1]]:

                props = dict(facecolor='white', alpha=0.0)

                RS = RectangleSelector(self.ax[1], self.select_callback_lens, useblit=True,
                                        minspanx=1, minspany=1,
                                        button=[3], spancoords='pixels',
                                        interactive=True, props=props)

                self.fig.canvas.mpl_connect('key_press_event', RS)
                plt.gcf().canvas.draw_idle()
                #

    def update_td_plot(self, beta1, beta2):

        #geomtd = self.geom_tdelay_(beta1, beta2)
        #gravtd = - self.lens.pot

        print (self.lens.zs)

        FOV = self.lens.theta1.max() - self.lens.theta1.min()

        if self.showtd:

            self.td = self.lens.t_delay_surf(beta=(beta1,beta2))#(geomtd + gravtd) * self.lens.conv_fact_time.value
            td_min = self.td.min()
            self.td -= td_min
            # define some contour levels
            max_td = 1.5 * ((-self.lens.pot).max()*self.lens.conv_fact_time.value - td_min)

            if max_td > 0:
                self.levels = np.linspace(0, max_td, 30)
            else:
                self.levels = 30

            self.ax[1].contourf(self.td, levels=self.levels, cmap=cm.coolwarm,
                                extent=[-FOV / 2., FOV / 2., -FOV / 2., FOV / 2.])
            self.ax[1].contour(self.td, levels=self.levels, extent=[-FOV / 2., FOV / 2., -FOV / 2., FOV / 2.],
                                colors='white')
            self.ax[0].contour(self.td, levels=self.levels, extent=[-FOV / 2., FOV / 2., -FOV / 2., FOV / 2.],
                                colors='white', alpha=0.0)

        tancau = self.lens.getCaustics(self.tancl)
        radcau = self.lens.getCaustics(self.radcl)

        for cl in self.tancl:
            thetac1, thetac2 = self.lens.getCritPoints(cl)
            self.ax[1].plot(thetac1, thetac2, '-', color='black', zorder=20)

        for cl in tancau:
            betac1, betac2 = self.lens.getCausticPoints(cl)
            self.ax[0].plot(betac1, betac2, '-', color='black', zorder=20)

        for cl in self.radcl:
            thetac1, thetac2 = self.lens.getCritPoints(cl)
            self.ax[1].plot(thetac1, thetac2, '-', color='black', zorder=20)

        for cl in radcau:
            betac1, betac2 = self.lens.getCausticPoints(cl)
            self.ax[0].plot(betac1, betac2, '-', color='black', zorder=20)


        if self.sersic:
            if isinstance(beta1, float):
                kwargs_se = {
                    'n': 1.0,
                    're': self.re,
                    'q': 1.0,
                    'pa': 0.0,
                    'ys1': beta1,
                    'ys2': beta2,
                    'zs': self.lens.zs
                }
            else:
                kwargs_se = {
                    'n': 1.0,
                    're': self.re,
                    'q': 1.0,
                    'pa': 0.0,
                    'ys1': beta1[0],
                    'ys2': beta2[0],
                    'zs': self.lens.zs
                }

            se = sersic(size=FOV, sizex=None, sizey=None, Npix=self.lens.nray1, gl=self.lens, save_unlensed=True,
                        **kwargs_se)

            if self.showtd:
                self.ax[0].imshow(se.image_unlensed, origin='lower', cmap='gray_r', alpha=0.7, zorder=30,
                                  extent=[-FOV / 2., FOV / 2., -FOV / 2., FOV / 2.])
                self.ax[1].imshow(se.image, origin='lower', cmap='gray_r', alpha=0.7, zorder=30,
                                  extent=[-FOV / 2., FOV / 2., -FOV / 2., FOV / 2.])
            else:
                self.ax[0].imshow(se.image_unlensed, origin='lower', cmap='afmhot_r', alpha=0.7, zorder=30,
                                extent=[-FOV / 2., FOV / 2., -FOV / 2., FOV / 2.])
                self.ax[1].imshow(se.image, origin='lower', cmap='afmhot_r', alpha=0.7, zorder=30,
                                extent=[-FOV / 2., FOV / 2., -FOV / 2., FOV / 2.])
            print (("Source position: %8.4f, %8.4f") % (beta1,beta2))

            if self.with_im_pos:
                kwargs_psr = {
                    'zs': self.lens.zs,
                    'ys1': beta1,
                    'ys2': beta2,
                    'flux': 1.0
                }
                ps = pointsrc(size=FOV, sizex=None, sizey=None, Npix=self.lens.nray1, gl=self.lens, **kwargs_psr)
                thetai_1, thetai_2, mui = ps.xi1, ps.xi2, ps.mui
                print("Image positions and magnifications:")
                for i in range(len(thetai_1)):
                    print(("-- image %i, theta=(%8.4f,%8.4f), mu=%8.4f") % (i, thetai_1[i], thetai_2[i], mui[i]))

                if len(thetai_1 > 0):
                    isel = mui > 25
                    mui[isel] = 25
                    if self.showtd:
                        self.ax[1].scatter(thetai_1, thetai_2, s=(mui * 5 + 15.0), marker='o', edgecolors='black',
                                           color='yellow')
                    else:
                        self.ax[1].scatter(thetai_1, thetai_2, s=(mui * 5 + 15.0), marker='o', edgecolors='black',
                                           color='red')

        else:
            self.ax[0].plot(beta1, beta2, 'o', ms=5, color='red')
            kwargs_psr = {
                'zs': self.lens.zs,
                'ys1': beta1,
                'ys2': beta2,
                'flux': 1.0
            }

            ps = pointsrc(size=FOV, sizex=None, sizey=None, Npix=self.lens.nray1, gl=self.lens, **kwargs_psr)
            thetai_1, thetai_2, mui = ps.xi1, ps.xi2, ps.mui
            print (("Source position: %8.4f, %8.4f") % (beta1,beta2))
            print ("Image positions and magnifications:")
            for i in range(len(thetai_1)):
                print (("-- image %i, theta=(%8.4f,%8.4f), mu=%8.4f") % (i,thetai_1[i],thetai_2[i],mui[i]))

            if len(thetai_1 > 0):
                isel = mui > 25
                mui[isel] = 25
                if self.showtd:
                    self.ax[1].scatter(thetai_1, thetai_2, s=(mui * 5 + 15.0), marker='o', edgecolors='black',
                                    color='yellow')
                else:
                    self.ax[1].scatter(thetai_1, thetai_2, s=(mui * 5 + 15.0), marker='o', edgecolors='black',
                                       color='red')
        #self.ax[0].set_xlim([-FOV / 2., FOV / 2.])
        #self.ax[0].set_ylim([-FOV / 2., FOV / 2.])
        #self.ax[1].set_xlim([-FOV / 2., FOV / 2.])
        #self.ax[1].set_ylim([-FOV / 2., FOV / 2.])
        self.ax[0].set_xlim(self.beta1_lim)
        self.ax[0].set_ylim(self.beta2_lim)
        self.ax[1].set_xlim(self.theta1_lim)
        self.ax[1].set_ylim(self.theta2_lim)
        plt.gcf().canvas.draw_idle()

    def select_callback_source(self, eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.beta1_lim = [min(x1, x2), max(x1, x2)]
        self.beta2_lim = [min(y1, y2), max(y1, y2)]
        self.ax[0].clear()
        self.ax[1].clear()
        self.update_td_plot(self.beta1, self.beta2)
        for i in range(2):
            self.ax[i].set_aspect('equal')
            self.ax[i].xaxis.set_tick_params(labelsize=fontsize)
            self.ax[i].yaxis.set_tick_params(labelsize=fontsize)
        self.ax[0].set_xlabel(r'$\beta_1$', fontsize=fontsize)
        self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
        self.ax[1].set_xlabel(r'$\theta_1$', fontsize=fontsize)
        self.ax[1].set_ylabel(r'$\theta_2$', fontsize=fontsize)

    def select_callback_lens(self, eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.theta1_lim = [min(x1, x2), max(x1, x2)]
        self.theta2_lim = [min(y1, y2), max(y1, y2)]
        self.ax[0].clear()
        self.ax[1].clear()
        self.update_td_plot(self.beta1, self.beta2)
        for i in range(2):
            self.ax[i].set_aspect('equal')
            self.ax[i].xaxis.set_tick_params(labelsize=fontsize)
            self.ax[i].yaxis.set_tick_params(labelsize=fontsize)
        self.ax[0].set_xlabel(r'$\beta_1$', fontsize=fontsize)
        self.ax[0].set_ylabel(r'$\beta_2$', fontsize=fontsize)
        self.ax[1].set_xlabel(r'$\theta_1$', fontsize=fontsize)
        self.ax[1].set_ylabel(r'$\theta_2$', fontsize=fontsize)

if __name__ == '__main__':
    cluster = ['S1063', 'M0416', 'M1206pl', 'PSZ1G311_200',
               'A370', 'A2744', 'M0717', 'M1149',
               'M0329', 'M1931', 'M2129', 'R2129']

    # create a dictionary with the cluster names and the corresponding number

    cluster_dict = {}
    for i in range(len(cluster)):
        cluster_dict[cluster[i]] = i

    parser = argparse.ArgumentParser(description='Interactive Lens VISualization Tool')
    parser.add_argument("-n", "--number", type=int, default=0,
                        help="cluster number. Options are: 0=S1063, 1=M0416, 2=M1206pl, 3=PSZ1G311_200, 4=A370, 5=A2744, 6=M0717, 7=M1149, 8=M0329, 9=M1931, 10=M2129, 11=R2129")
    parser.add_argument("-z", "--source_redshift", type=float, default=1.0, help="source redshift")
    parser.add_argument("-r", "--source_re", type=float, default=1.0, help="source effective radius")

    args = parser.parse_args()
    icl = args.number
    zs = args.source_redshift
    re = args.source_re

    # posizione delle mappe degli angoli di deflessione
    path_to_angles = ['/Users/massimo/stiva/pietro_models/',
                      '/Users/massimo/stiva/pietro_models/',
                      '/Users/massimo/stiva/pietro_models/',
                      '/Users/massimo/stiva/clusters/PSZ1G311/delens/',
                      '/Users/massimo/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                      '/Users/massimo/stiva/pietro_models/',
                      '/Users/massimo/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                      '/Users/massimo/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
                      '/Users/massimo/stiva/gabriel_models/best_fit/',
                      '/Users/massimo/stiva/gabriel_models/best_fit/',
                      '/Users/massimo/stiva/gabriel_models/best_fit/',
                      '/Users/massimo/stiva/gabriel_models/best_fit/']

    zl_s = lst.getLensRedshift(path_to_angles[icl] + cluster[icl] + '.par')
    print(path_to_angles[icl] + cluster[icl] + '.par')
    print(('Working with cluster %s at redshift %s') % (cluster[icl], zl_s))
    zl = float(zl_s)

    print('creating deflector...')
    df = lst.create_deflector(parfile=path_to_angles[icl] + cluster[icl] + '.par',
                              filex=path_to_angles[icl] + cluster[icl] + '_angx.fits',
                              filey=path_to_angles[icl] + cluster[icl] + '_angy.fits',
                              zl=zl, zs=zs, zsnorm=1.0, resc_fact=1.0, compute_potential=True)

    lims = lst.getFoV(path_to_angles[icl] + cluster[icl] + '.par')

    beta1_lim = [lims[0], lims[1]]
    beta2_lim = [lims[2], lims[3]]

    import matplotlib.pyplot as plt

    kwargs_plot = {'beta1_lim': beta1_lim,
                   'beta2_lim': beta2_lim,
                   'theta1_lim': beta1_lim,
                   'theta2_lim': beta2_lim,
                   'cluster': cluster,
                   'path': path_to_angles}
    li = lens_interact(image=None, sersic=True, re=re, with_im_pos=True, **kwargs_plot)
    plt.show()
