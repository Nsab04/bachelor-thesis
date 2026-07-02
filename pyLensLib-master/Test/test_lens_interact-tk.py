import pyLensLib.lenstool as lst
from pyLensLib.lens_interact import lens_interact
from  tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

cluster = ['S1063','M0416','M1206pl','PSZ1G311_200',
           'A370','A2744','M0717','M1149',
           'M0329','M1931','M2129','R2129']

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

def main():
    root = Tk()
    gui = Window(root)
    gui.root.mainloop()
    return None

class Window:
    def __init__(self,root):
        self.root = root
        self.root.title("Lens interactive tool")
        self.root.geometry('1500x1000')

        # cluster number
        self.icl = 0
        Label(self.root, text="Cluster number").grid(row=0, column=0)
        self.icl_entry = Entry(self.root, width=5)
        self.icl_entry.grid(row=0, column=1)

        # source redshift
        self.zs = 1.0
        Label(self.root, text="Source redshift").grid(row=1, column=0)
        self.zs_entry = Entry(self.root, width=5)
        self.zs_entry.grid(row=1, column=1)

        # effective radius
        self.re = 0.5
        Label(self.root, text="Effective radius").grid(row=2, column=0)
        self.re_entry = Entry(self.root, width=5)
        self.re_entry.grid(row=2, column=1)

        # Update Button
        button1 = Button(self.root, text="Calculate", command=self.update_values)
        button1.grid(row=13, column=0)
        self.root.bind("<Return>", self.update_values)

        #print (self.icl,self.re,self.zs)
        self.make_plot()
        pass

    def update_values(self, event=None):
        self.icl = int(self.icl_entry.get())
        self.zs = float(self.zs_entry.get())
        self.re = float(self.re_entry.get())
        #print (self.icl,self.re,self.zs)
        self.make_plot()
        return None

    def make_plot(self):
        zl_s = lst.getLensRedshift(path_to_angles[self.icl] + cluster[self.icl] + '.par')
        zl = float(zl_s)
        df = lst.create_deflector(parfile=path_to_angles[self.icl] + cluster[self.icl] + '.par',
                                  filex=path_to_angles[self.icl] + cluster[self.icl] + '_angx.fits',
                                  filey=path_to_angles[self.icl] + cluster[self.icl] + '_angy.fits',
                                  zl=zl, zs=self.zs, zsnorm=1.0, resc_fact=1.0, compute_potential=True)
        kwargs_plot = {'beta1_lim': [-20, 20], 'beta2_lim': [-20, 20], 'theta1_lim': [-60, 60], 'theta2_lim': [-60, 60]}
        li = lens_interact(df, image=None, sersic=True, re=self.re, with_im_pos=True, **kwargs_plot)
        #plt.show()
        chart = FigureCanvasTkAgg(li.fig, self.root)
        chart.get_tk_widget().grid(row=0, column=3)

main()

