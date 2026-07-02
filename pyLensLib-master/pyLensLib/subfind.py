import pyLensLib.gadget as G
import g3read
import pandas as pd
#from pysimlib.gadget import snapshot_header
import numpy as np
import matplotlib.pyplot as plt
#import scipy.ndimage as ndimage
#from matplotlib.colors import LogNorm
#from astropy.io import fits
#from astropy.cosmology import FlatLambdaCDM



class subfind(object):
    """
    Subhalo finder and analysis class for simulation data.

    Attributes:
        sc: Snapshot header or file header object.
        mass, mvir, rvir, nsub: Main halo properties.
        fsub, msub, vmax, rmax, rhalf, dsub: Subhalo properties.
        gpos, spos: Positions of halos and subhalos.
        grnr: Group number for subhalos.
        smtab: Subhalo mass table.
        pid: Particle IDs (if with_pid=True).
        soff, slen: Subhalo offsets and lengths.
        redshift: Simulation redshift.
    """
    
    def __init__(self,filen,with_pid=False):
        """
        Initialize subfind object and load halo/subhalo data from file.

        Args:
            filen (str): Filename of the simulation snapshot or catalog.
            with_pid (bool, optional): If True, load particle IDs.

        Returns:
            None
        """

        # main halos
        #self.sc=G.snapshot_header(filen)
        self.sc = g3read.GadgetFile(filen,is_snap=False).header

        self.redshift=self.sc.redshift
        fof = g3read.read_new(filen,['MTOT','MVIR','RVIR','NSUB',
                                     'FSUB','MSUB','VMAX','RMAX',
                                     'RHMS','DSUB','GPOS','GRNR',
                                     'SPOS'],0,is_snap=False)
        #self.mass=G.read_block(filen,'MTOT')
        self.mass = fof['MTOT']
        #self.mvir=G.read_block(filen,'MVIR')
        self.mvir = fof['MVIR']
        #self.rvir=G.read_block(filen,'RVIR')
        self.rvir = fof['RVIR']
        self.rvir=self.rvir/1000.0
        #self.nsub=G.read_block(filen,'NSUB')
        self.nsub = fof['NSUB']
        self.mass=self.mass*1e10
        self.mvir=self.mvir*1e10

        # subhalos
        #self.fsub=G.read_block(filen,'FSUB')
        self.fsub = fof['FSUB']
        #self.msub=G.read_block(filen,'MSUB')
        self.msub = fof['MSUB']
        #self.vmax=G.read_block(filen,'VMAX')
        self.vmax = fof['VMAX']
        #self.rmax=G.read_block(filen,'RMAX')
        self.rmax = fof['RMAX']
        #self.rhalf=G.read_block(filen,'RHMS')
        self.rhalf = fof['RHMS']
        #self.dsub=G.read_block(filen,'DSUB')
        self.dsub = fof['DSUB']
        #self.gpos=G.read_block(filen,'GPOS')
        self.gpos = fof['GPOS']
        self.grnr=G.read_block(filen,'GRNR')
        #self.grnr = fof['GRNR']
        self.spos=G.read_block(filen,'SPOS')
        #self.spos = fof['SPOS']
        #print ('fof',fof)

        if min(self.spos[:, 0]) < -4e5:
            mask = (self.spos < 0)

            self.spos[mask] = self.spos[mask] + self.sc.boxsize / 2.
            self.spos[~mask] = self.spos[~mask] - self.sc.boxsize / 2.

        if min(self.gpos[:, 0]) < -4e5:
            mask = (self.gpos < 0)

            self.gpos[mask] = self.gpos[mask] + self.sc.boxsize / 2.
            self.gpos[~mask] = self.gpos[~mask] - self.sc.boxsize / 2.

        self.gpos = self.gpos / 1000.0
        self.spos=self.spos/1000.0
        self.smtab=G.read_block(filen,'SMST')
        if with_pid:
            self.pid=G.read_block(filen,'PID')
        self.soff=G.read_block(filen,'SOFF')
        self.slen=G.read_block(filen,'SLEN')

    def plot_shmf(self,ax=None,minmass=1e10):
        """
        Plot the subhalo mass function (SHMF) for the most massive halo.

        Args:
            ax: Matplotlib axis to plot on.
            minmass (float, optional): Minimum halo mass to consider.

        Returns:
            None
        """
        #locate the halo with the maximum mass in the catalog:
        index_max = np.argmax(self.mvir)
        
        if (self.mvir[index_max]<minmass):
            return
    
        isel= self.grnr == index_max
        x=(self.spos[isel,0]-self.gpos[index_max,0])/(1.0+self.redshift)
        y=(self.spos[isel,1]-self.gpos[index_max,1])/(1.0+self.redshift)
        z=(self.spos[isel,2]-self.gpos[index_max,2])/(1.0+self.redshift)
        
        ms_=self.msub[isel]*1e10/self.mvir[index_max]
        ii = ((np.sqrt(x**2 + y**2 + z**2) < self.rvir[index_max]/(1.0+self.redshift)) & 
              (self.msub[isel]*1e10 < self.mvir[index_max]*0.1) &
              (self.smtab[isel,1] > 0))
        
        if (ii.sum()>0):
            ms=ms_[ii]
            print ('Halo Mvir: %10e | N of SubHalos: %i | SH min. mass: %10e | tot. SH mass: %10e | fsub: %4.2f' % 
                   (self.mvir[index_max], 
                    self.nsub[index_max],
                    self.msub[self.grnr == index_max].min()*1e10, 
                    self.msub[self.grnr == index_max].sum()*1e10-self.msub[self.grnr == index_max].max()*1e10, 
                    ms.sum()-ms.max()))

            nbins=10
            dn,b = np.histogram(np.log10(ms),nbins)
            x = (b[1:] + b[:-1])/2.0
            dx = b[1]-b[0]
            dn = dn/dx#/self.mvir[index_max]
            ax.plot(10**x,dn,color='grey',linestyle='-',alpha=0.3)

        #h,bins,patches=ax.hist(np.log10(ms),bins=40,alpha=0.5,histtype='step',normed=True)
        #xb=[0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)]
        #ax.plot(xb,h)
        #sns.distplot(np.log10(ms*1e10),bins=40,hist_kws={'log':True,'histtype':'step'},ax=ax)
        
        
    def plot_raddist(self,ax=None,minmass=1e10):
        """
        Plot the radial distribution of subhalos in 3D for the most massive halo.

        Args:
            ax: Matplotlib axis to plot on.
            minmass (float, optional): Minimum halo mass to consider.

        Returns:
            None
        """
 
        #locate the halo with the maximum mass in the catalog:
        index_max = np.argmax(self.mvir)
        
        if (self.mvir[index_max]<minmass):
            return
    
        isel= self.grnr == index_max
        x=(self.spos[isel,0]-self.gpos[index_max,0])/(1.0+self.redshift)
        y=(self.spos[isel,1]-self.gpos[index_max,1])/(1.0+self.redshift)
        z=(self.spos[isel,2]-self.gpos[index_max,2])/(1.0+self.redshift)
        
        d_=np.sqrt(x**2 + y**2 + z**2)
        
        ii = ((d_ < self.rvir[index_max]/(1.0+self.redshift)) & 
              (self.msub[isel]*1e10 < self.mvir[index_max]*0.1) &
              (self.smtab[isel,1] > 0))
        
        if (ii.sum()>0):
            d=d_[ii]/self.rvir[index_max]*(1.0+self.redshift)
            nbins=30
            dn,b = np.histogram(np.log10(d),nbins)
            x = (b[1:] + b[:-1])/2.0
            dx = b[1]-b[0]
            ncum = np.array([np.sum(dn[i:]) for i in range(len(dn))]).astype('float')
            ax.plot(10**x,ncum/np.sum(dn),color='grey',linestyle='-',alpha=0.3)
        
    def plot_cumshmf(self,ax=None,minmass=1e10):
        """
        Plot the cumulative subhalo mass function for the most massive halo.

        Args:
            ax: Matplotlib axis to plot on.
            minmass (float, optional): Minimum halo mass to consider.

        Returns:
            None
        """
        #locate the halo with the maximum mass in the catalog:
        index_max = np.argmax(self.mvir)
        
        if (self.mvir[index_max]<minmass):
            return
    
        isel= self.grnr == index_max
        x=(self.spos[isel,0]-self.gpos[index_max,0])/(1.0+self.redshift)
        y=(self.spos[isel,1]-self.gpos[index_max,1])/(1.0+self.redshift)
        z=(self.spos[isel,2]-self.gpos[index_max,2])/(1.0+self.redshift)
        
        ms_=self.msub[isel]*1e10
        ii = ((np.sqrt(x**2 + y**2 + z**2) < self.rvir[index_max]/(1.0+self.redshift)) &
              (self.smtab[isel,1] > 0))
        
        
        
        if (ii.sum()>0):
            ms=ms_[ii]
            nbins=30
            dn,b = np.histogram(np.log10(ms),nbins)
            x = (b[1:] + b[:-1])/2.0
            dx = b[1]-b[0]
            ncum = [np.sum(dn[i:]) for i in range(len(dn))]
            ax.plot(10**x,ncum,color='grey',linestyle='-',alpha=0.3)


        
    def plot_vcm(self,ax=None,minmass=1e10,color=None,marker='o',cmap='viridis'):
        """
        Plot max circular velocity vs mass of subhalos within the virial radius.

        Args:
            ax: Matplotlib axis to plot on.
            minmass (float, optional): Minimum halo mass to consider.
            color: Color for plot points.
            marker (str, optional): Marker style.
            cmap (str, optional): Colormap for scatter plot.

        Returns:
            pandas.DataFrame: Dataframe with msub, vmax, zl, and stype.
        """
 
        #locate the halo with the maximum mass in the catalog:
        index_max = np.argmax(self.mvir)
        
        if (self.mvir[index_max]<minmass):
            return
    
        isel= self.grnr == index_max
        x=(self.spos[isel,0]-self.gpos[index_max,0])/(1.0+self.redshift)
        y=(self.spos[isel,1]-self.gpos[index_max,1])/(1.0+self.redshift)
        z=(self.spos[isel,2]-self.gpos[index_max,2])/(1.0+self.redshift)
        d_=np.sqrt(x**2 + y**2 + z**2)
        
        ms_=self.msub[isel]*1e10#/self.mvir[index_max]
        vcirc_=self.vmax[isel]
        ii = ((d_ < self.rvir[index_max]/(1.0+self.redshift)) &
              (self.smtab[isel,1] > 0))
        
        if (ii.sum()>0):
            ms=ms_[ii]
            vcirc=vcirc_[ii]
            d=d_[ii]/self.rvir[index_max]*(1.0+self.redshift)
            #print ('Halo Mvir: %10e | N of SubHalos: %i | SH min. mass: %10e | tot. SH mass: %10e | fsub: %4.2f' % 
            #       (self.mvir[index_max], 
            #        self.nsub[index_max],
            #        self.msub[self.grnr == index_max].min()*1e10, 
            #        self.msub[self.grnr == index_max].sum()*1e10-self.msub[self.grnr == index_max].max()*1e10, 
            #        ms.sum()-ms.max()))
            if (color==None):
                ax.scatter(ms,vcirc,marker=marker,c=d,cmap=cmap)
            else:
                ax.plot(ms,vcirc,'o',alpha=0.3,color=color)
            stype=['AGN' for i in range(len(ms))]
            data = {'msub': np.log10(ms), 'vmax': vcirc, 'zl': self.redshift, 'stype': stype} 
            df = pd.DataFrame(data,columns=['msub','vmax','zl','stype'])
            return(df)
        
    def get_dataframe(self,minmass=1e10,proj=0,Rmax=0.16,msub_min=1e10):
        """
        Return a dataframe with subhalo properties for the most massive halo.

        Args:
            minmass (float, optional): Minimum halo mass to consider.
            proj (int, optional): Projection axis (0, 1, or 2).
            Rmax (float, optional): Maximum projected radius (fraction of rvir).
            msub_min (float, optional): Minimum subhalo mass.

        Returns:
            pandas.DataFrame: Dataframe with msub, mstar, vmax, zl, Rnorm, stype.
        """
 
        #locate the halo with the maximum mass in the catalog:
        index_max = np.argmax(self.mvir)

        # set-up projections
        if proj == 0:
            ip=0
            jp=1
            kp=2
        elif proj == 1:
            ip=0
            jp=2
            kp=1
        else:
            ip=1
            jp=2
            kp=0

        
        if (self.mvir[index_max]<minmass):
            return
    
        isel= self.grnr == index_max
        x=(self.spos[isel,ip]-self.gpos[index_max,ip])/(1.0+self.redshift)
        y=(self.spos[isel,jp]-self.gpos[index_max,jp])/(1.0+self.redshift)
        z=(self.spos[isel,kp]-self.gpos[index_max,kp])/(1.0+self.redshift)
        
        ms_=self.msub[isel]*1e10#/self.mvir[index_max]
        vcirc_=self.vmax[isel]
        mstar_=self.smtab[isel,4]*1e10
        
        
        dist_=np.sqrt(x**2 + y**2)
        ii = ((np.sqrt(x**2 + y**2) < Rmax*self.rvir[index_max]/(1.0+self.redshift)) & 
              (np.sqrt(z**2) < self.rvir[index_max]/(1.0+self.redshift)) & 
              (self.msub[isel]*1e10 < self.mvir[index_max]*0.1) &
              (self.msub[isel]*1e10 > msub_min) &
              (self.smtab[isel,1] > 0))
        
        if (ii.sum()>0):
            ms=ms_[ii]
            vcirc=vcirc_[ii]
            dist=dist_[ii]/self.rvir[index_max]*(1.0+self.redshift)
            mstar=mstar_[ii]
            stype=['AGN' for i in range(len(ms))]
            data = {'msub': np.log10(ms), 'mstar': np.log10(mstar), 'vmax': vcirc, 'zl': self.redshift, 'Rnorm': dist, 'stype': stype}
            df = pd.DataFrame(data,columns=['msub','mstar','vmax','zl', 'Rnorm', 'stype'])
            return(df)

    def plot_rmaxm(self,ax=None,minmass=1e10):
        """
        Plot subhalo mass vs half-mass radius for the most massive halo.

        Args:
            ax: Matplotlib axis to plot on.
            minmass (float, optional): Minimum halo mass to consider.

        Returns:
            None
        """
 
        #locate the halo with the maximum mass in the catalog:
        index_max = np.argmax(self.mvir)
        
        if (self.mvir[index_max]<minmass):
            return
    
        isel= self.grnr == index_max
        x=(self.spos[isel,0]-self.gpos[index_max,0])/(1.0+self.redshift)
        y=(self.spos[isel,1]-self.gpos[index_max,1])/(1.0+self.redshift)
        z=(self.spos[isel,2]-self.gpos[index_max,2])/(1.0+self.redshift)
        
        ms_=self.msub[isel]*1e10
        rhalf_=self.rhalf[isel]
        ii = ((np.sqrt(x**2 + y**2 + z**2) < self.rvir[index_max]/(1.0+self.redshift)) &
              (self.smtab[isel,1] > 0))
        
        if (ii.sum()>0):
            ms=ms_[ii]
            rhalf=rhalf_[ii]
            ax.plot(ms,rhalf,'o',alpha=0.3)
            
    def plot_sigmavc(self,ax=None,minmass=1e10):
        """
        Plot velocity dispersion vs circular velocity for subhalos.

        Args:
            ax: Matplotlib axis to plot on.
            minmass (float, optional): Minimum halo mass to consider.

        Returns:
            None
        """

        #locate the halo with the maximum mass in the catalog:
        index_max = np.argmax(self.mvir)
        
        if (self.mvir[index_max]<minmass):
            return
    
        isel= self.grnr == index_max
        x=(self.spos[isel,0]-self.gpos[index_max,0])/(1.0+self.redshift)
        y=(self.spos[isel,1]-self.gpos[index_max,1])/(1.0+self.redshift)
        z=(self.spos[isel,2]-self.gpos[index_max,2])/(1.0+self.redshift)
        
        vmax_=self.vmax[isel]
        sigma_=self.dsub[isel]
        ii = ((np.sqrt(x**2 + y**2 + z**2) < self.rvir[index_max]/(1.0+self.redshift)) &
              (self.smtab[isel,1] > 0))
        
        if (ii.sum()>0):
            vmax=vmax_[ii]
            sigma=sigma_[ii]

            ax.plot(vmax,sigma,'o',alpha=0.3)

    def plot_pshmf(self,ax=None,minmass=1e10,depth=10.0,squared=True,proj=0,Rmax=0.16):
        """
        Plot the projected cumulative mass function of subhalos within projected radius.

        Args:
            ax: Matplotlib axis to plot on.
            minmass (float, optional): Minimum halo mass to consider.
            depth (float, optional): Depth for selection.
            squared (bool, optional): Use squared selection region.
            proj (int, optional): Projection axis (0, 1, or 2).
            Rmax (float, optional): Maximum projected radius (fraction of rvir).

        Returns:
            None
        """

        #locate the halo with the maximum mass in the catalog:
        index_max = np.argmax(self.mvir)
        
        # set-up projections
        if proj == 0:
            ip=0
            jp=1
            kp=2
        elif proj == 1:
            ip=0
            jp=2
            kp=1
        else:
            ip=1
            jp=2
            kp=0
        # measure the projected distance of each group from the most massive group
        dist_halo=np.sqrt((self.gpos[:,ip]-self.gpos[index_max,ip])**2+(self.gpos[:,jp]-self.gpos[index_max,jp])**2)
        
        #Select all halos within a distance of 5*sidel
        sel_halos=dist_halo < 5.0
        
        # if the halo is too small, throw it away
        if (self.mvir[index_max]<minmass):
            return

        idone=False

        # loop over the groups
        for i in [index_max]:#range(len(self.mass)):
            # if the group is close enough...
            sel_halos[i]=True
            if (sel_halos[i]):
                # ...select all its subhalos
                isel= self.grnr == i
                
                # ...and compute their positions with respect to the new origin (physical)
                x=(self.spos[isel,ip]-self.gpos[index_max,ip])/(1.0+self.redshift)
                y=(self.spos[isel,jp]-self.gpos[index_max,jp])/(1.0+self.redshift)
                z=(self.spos[isel,kp]-self.gpos[index_max,kp])/(1.0+self.redshift)
                
                
                # normalize the subhalo masses by the virial mass of the host
                ms_=self.msub[isel]*1e10
                #print ('Min mass of subhalos: %e' % (ms_.min()))
                #print('%e %e' % (self.smtab[isel,:].sum()*1e10,ms_.sum()))

                if squared:
                    ii = ((np.abs(x) < self.rvir[index_max]/(1.0+self.redshift)) &
                          (np.abs(y) < self.rvir[index_max]/(1.0+self.redshift)) &
                          (np.abs(z) < self.rvir[index_max]/(1.0+self.redshift)))
                else:
                    ii = ((np.sqrt(x**2 + y**2) < Rmax*self.rvir[index_max]/(1.0+self.redshift)) & 
                          (np.sqrt(z**2) < self.rvir[index_max]/(1.0+self.redshift)) &
                          (self.smtab[isel,1] > 0) & 
                          (self.msub[isel]*1e10 < self.mvir[index_max]*0.1))

                    
                if (ii.sum()>0):
                    #print ('group %i nsub sel %i' % (i,np.sum(ii)))
                    if (not idone):
                        ms=ms_[ii]
                        idone=True
                    else:
                        ms=np.concatenate((ms,ms_[ii]),axis=0)
        
        

        if not idone:
            return

        #print ('Mass in subs: %e' % (ms.sum()-ms.max()))
        nbins=160
        dn,b = np.histogram(np.log10(ms),nbins)
        x = (b[1:] + b[:-1])/2.0
        dx = b[1]-b[0]
        ncum = [np.sum(dn[i:]) for i in range(len(dn))]
        ax.plot(10**x,ncum,color='grey',linestyle='-',alpha=0.3)
        #print ("Min mass of subhalos %e " % (np.min(ms)))
        

        #h,bins,patches=ax.hist(np.log10(ms),bins=40,alpha=0.5,histtype='step',normed=True)
        #xb=[0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)]
        #ax.plot(xb,h)
        #sns.distplot(np.log10(ms*1e10),bins=40,hist_kws={'log':True,'histtype':'step'},ax=ax)        
 

    def plot_vcirc(self,ax=None,minmass=1e10,depth=10.0,squared=True,proj=0,Rmax=0.16,msub_min=1e10):
        """
        Plot the cumulative distribution of circular velocities for subhalos.

        Args:
            ax: Matplotlib axis to plot on.
            minmass (float, optional): Minimum halo mass to consider.
            depth (float, optional): Depth for selection.
            squared (bool, optional): Use squared selection region.
            proj (int, optional): Projection axis (0, 1, or 2).
            Rmax (float, optional): Maximum projected radius (fraction of rvir).
            msub_min (float, optional): Minimum subhalo mass.

        Returns:
            None
        """
        #locate the halo with the maximum mass in the catalog:
        index_max = np.argmax(self.mvir)
        
        # set-up projections
        if proj == 0:
            ip=0
            jp=1
            kp=2
        elif proj == 1:
            ip=0
            jp=2
            kp=1
        else:
            ip=1
            jp=2
            kp=0
        # measure the projected distance of each group from the most massive group
        dist_halo=np.sqrt((self.gpos[:,ip]-self.gpos[index_max,ip])**2+(self.gpos[:,jp]-self.gpos[index_max,jp])**2)
        
        #Select all halos within a distance of 5*sidel
        sel_halos=dist_halo < 5.0
        
        # if the halo is too small, throw it away
        if (self.mvir[index_max]<minmass):
            return

        idone=False

        # loop over the groups
        for i in [index_max]:#range(len(self.mass)):
            # if the group is close enough...
            sel_halos[i]=True
            if (sel_halos[i]):
                # ...select all its subhalos
                isel= self.grnr == i
                
                # ...and compute their positions with respect to the new origin (physical)
                x=(self.spos[isel,ip]-self.gpos[index_max,ip])/(1.0+self.redshift)
                y=(self.spos[isel,jp]-self.gpos[index_max,jp])/(1.0+self.redshift)
                z=(self.spos[isel,kp]-self.gpos[index_max,kp])/(1.0+self.redshift)
                
                vcirc_=self.vmax[isel]
                
                if squared:
                    ii = ((np.abs(x) < self.rvir[index_max]/(1.0+self.redshift)) &
                          (np.abs(y) < self.rvir[index_max]/(1.0+self.redshift)) &
                          (np.abs(z) < self.rvir[index_max]/(1.0+self.redshift)))
                else:
                    ii = ((np.sqrt(x**2 + y**2) < Rmax*self.rvir[index_max]/(1.0+self.redshift)) & 
                          (np.sqrt(z**2) < self.rvir[index_max]/(1.0+self.redshift)) & 
                          (self.msub[isel]*1e10 < self.mvir[index_max]*0.1) &
                          (self.msub[isel]*1e10 > msub_min) &
                          (self.smtab[isel,1] > 0))
                    
                if (ii.sum()>0):
                    #print ('group %i nsub sel %i' % (i,np.sum(ii)))
                    if (not idone):
                        vcirc=vcirc_[ii]
                        idone=True
                    else:
                        vcirc=np.concatenate((vcirc,vcirc_[ii]),axis=0)
        
                    nbins=160
                    dn,b = np.histogram(vcirc,nbins)
                    x = (b[1:] + b[:-1])/2.0
                    dx = b[1]-b[0]
                    ncum = [np.sum(dn[i:]) for i in range(len(dn))]
                    ax.plot(x,ncum,color='grey',linestyle='-',alpha=0.3)
        
    def plot_sp(self,filename='test.png',proj=0):
        """
        Plot the spatial distribution of halos and subhalos in projection.

        Args:
            filename (str, optional): Output filename for plot image.
            proj (int, optional): Projection axis (0, 1, or 2).

        Returns:
            None
        """
    
        index_max = np.argmax(self.mvir)
        
        if proj == 0:
            ip=0
            jp=1
            kp=2
        elif proj == 1:
            ip=0
            jp=2
            kp=1
        else:
            ip=1
            jp=2
            kp=0

        fig,ax=plt.subplots(1,1,figsize=(10,10))
        ax.plot(self.gpos[:,ip],self.gpos[:,jp],'o',alpha=0.1)
        ax.plot(self.gpos[index_max,ip],self.gpos[index_max,jp],'o',color='red',zorder=10)

        #subhalos beloging to the most massive halo
        isel= self.grnr == index_max
        ax.plot(self.spos[isel,ip],self.spos[isel,jp],',',color='orange',zorder=5)

        #subhalos belonging to the second most massive halos
        #for i in range(1,10):
        #    isel= grnr == i
        #    ax.plot(gpos[i,0],gpos[i,1],'o',color='yellow',zorder=10)
        #    ax.plot(spos[isel,0],spos[isel,1],',',zorder=5)
        fig.savefig(filename)
