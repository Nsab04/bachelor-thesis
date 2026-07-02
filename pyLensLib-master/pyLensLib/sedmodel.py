import pandas as pd
import pyLensLib.sedcompat as ss
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
import astropy.constants as const
import warnings

c = const.c.value
Jy = 1e-23

class sedmodel(object):

    """
    Class for handling SED (Spectral Energy Distribution) templates and operations.

    Attributes:
        sed_dir (str): Directory containing SED templates.
        seds_df (DataFrame): DataFrame of available SED templates.
    """
    def __init__(self,sed_dir,sed_list):
        """
        Initialize a sedmodel by loading a list of SED templates.

        Args:
            sed_dir (str): Directory where the SED templates are stored.
            sed_list (str): File containing the list of SED templates available in sed_dir.

        Returns:
            None
        """
        self.sed_dir = sed_dir
        try:
            self.seds_df = pd.read_csv(sed_dir+'/'+sed_list, header=None, delim_whitespace=True)
            self.seds_df.columns=['sed']
        except Exception as exc:
            raise RuntimeError('Problem reading the sed template list') from exc

    def plotTemplates(self,save_to_file=None):
        """
        Plot all SED templates.

        Args:
            save_to_file (str, optional): If provided, save plot to file.

        Returns:
            None
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i in range(len(self.seds_df.sed)):
            l, f = np.loadtxt(self.sed_dir + '/'+ self.seds_df.sed[i], unpack=True)
            ax.plot(l, f, '-', label=self.seds_df.sed[i])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([800, 2e5])
        ax.set_ylim([2e-5, 200])
        ax.legend()
        lsize = 22
        ax.set_xlabel(r'$\lambda$ [Angstrom]', fontsize=lsize)
        ax.set_ylabel(r'$Flux$', fontsize=lsize)
        ax.xaxis.set_tick_params(labelsize=lsize)
        ax.yaxis.set_tick_params(labelsize=lsize)
        if save_to_file != None:
            fig.savefig(save_to_file)
        else:
            plt.show()

    def getSED(self,ised=0,z=0):
        """
        Return the SED template by index, optionally redshifted.

        Args:
            ised (int): Index of the SED template in the list.
            z (float): Redshift to apply (default 0).

        Returns:
            SED: Barak SED object.
        """
        l, f = np.loadtxt(self.sed_dir + '/' + self.seds_df.sed[ised], unpack=True)
        sed_0 = ss.SED(wa=l, fl=f, z=0)
        if z > 0:
            sed_z = sed_0.copy()
            sed_z.redshift_to(z)
            return sed_z
        else:
            return sed_0

    def getrSED(self,rsed=0,z=0):
        """
        Return an interpolated SED template between two indices, optionally redshifted.

        Args:
            rsed (float): Index for interpolation between templates.
            z (float): Redshift to apply (default 0).

        Returns:
            SED: Barak SED object.
        """
        nsed = len(self.seds_df.sed)
        if rsed < 0 or rsed > (nsed - 1):
            raise ValueError(f"rsed must be in [0, {nsed - 1}]")

        ised=int(rsed)
        if ised >= nsed - 1:
            return self.getSED(ised=nsed - 1, z=z)

        rf = rsed-ised
        l0, f0 = np.loadtxt(self.sed_dir + '/' + self.seds_df.sed[ised], unpack=True)
        l1, f1_ = np.loadtxt(self.sed_dir + '/' + self.seds_df.sed[ised+1], unpack=True)
        f1f = interp1d(l1, f1_, fill_value="extrapolate")  # ,kind='cubic')
        f1 = f1f(l0)
        f = f0 + (f1 - f0) * rf
        sed_0 = ss.SED(wa=l0, fl=f, z=0)
        if z > 0:
            sed_z = sed_0.copy()
            sed_z.redshift_to(z)
            return sed_z
        else:
            return sed_0


class clusterSEDs(object):

    """
    Class for handling SEDs of galaxy clusters, adapted from barak package.

    Attributes:
        homedir (str): Directory containing SED templates.
        sed_list (str): SED list filename.
        morphology (list): List of galaxy morphologies.
        mag_ref (ndarray): Reference magnitudes.
        templates_ell (list): Indices for elliptical templates.
        templates_sp (list): Indices for spiral templates.
        zhalo (float): Cluster redshift.
        ref_band (str): Reference bandpass file.
        wa0, fl0 (ndarray): Wavelengths and fluxes for SEDs.
        waz, flz (ndarray): Redshifted wavelengths and fluxes.
        waref, trref (ndarray): Reference bandpass data.
        refband_flambda (ndarray): Reference bandpass fluxes.
    """
    def __init__(self, **kwargs):
        if ('seddir' in kwargs):
            self.homedir = kwargs['seddir']
        else:
            self.homedir = '/home/lleuzzi/projects/CODES/bpz-1.99.3/SED/'
        if ('sed_list' in kwargs):
            self.sed_list = kwargs['sed_list']
        else:
            self.sed_list = 'eB11.list'
        if ('morphology' in kwargs):
            self.morphology = kwargs['morphology']
        else:
            self.morphology = ['elliptical']
        if ('mag_ref' in kwargs):
            self.mag_ref = np.array(kwargs['mag_ref'])
        else:
            self.mag_ref = np.array([23.])
        if ('templates_ell' in kwargs):
            self.templates_ell = kwargs['templates_ell']
        else:
            self.templates_ell = [0,1,2,3]
        if ('templates_sp' in kwargs):
            self.templates_sp = kwargs['templates_sp']
        else:
            self.templates_sp = [5,6]
        if ('zhalo' in kwargs):
            self.zhalo = kwargs['zhalo']
        else:
            self.zhalo = 0.5
        if ('reference_band' in kwargs):
            self.ref_band = kwargs['reference_band']
        else:
            self.ref_band = '/home/lleuzzi/projects/CODES/bpz-1.99.3/FILTER/HST_ACS_WFC_F606W.res'

        # let's import SED files
        sed_list = self.homedir + self.sed_list
        seds_df = pd.read_csv(sed_list, header=None, delim_whitespace=True)
        seds_df.columns = ['sed']

        wavelengths = []
        fluxes = []

        for k, mag_gal in enumerate(self.mag_ref):
            """
            MM: there are only 5 discrete SEDs for elliptical galaxies and 2 for other 
               morphological types. 
               As it is now, if the SED list is changed there is no coupling with the SED 
               types specified here!
            """

            if (self.morphology[k] == "elliptical"):
                sed_num = random.choice([seds_df.sed[i] for i in self.templates_ell])
            else:
                sed_num = random.choice([seds_df.sed[i] for i in self.templates_sp])

            # for each galaxy, read the sed template and append
            l, f = np.loadtxt(self.homedir + sed_num, unpack=True)
            fluxes.append(f)
            wavelengths.append(l)

        # store wavelength and flux in two numpy arrays
        self.wa0 = np.array(wavelengths, dtype=object)
        self.fl0 = np.array(fluxes, dtype=object)

        # compute the integral of each SED in the numpy array
        #z0fluxtot = np.array([np.trapz(sub_x, sub_y) for sub_x, sub_y in zip(self.wa0, self.fl0)])

        # shift the SEDs according to a common galaxy redshift
        waz = self.wa0 * (self.zhalo + 1)

        #zfluxtot = np.array([np.trapz(sub_x, sub_y) for sub_x, sub_y in zip(waz, self.fl0)])

        flz = self.fl0 / (self.zhalo + 1) #* (z0fluxtot / zfluxtot)


        self.waref, self.trref = self.passband(self.ref_band)

        # sed normalization
        self.refband_flambda = self.mag2flux(self.mag_ref, self.waref, self.trref)
        # print("f606_l = ", f606_flambda)
        sedflux = np.array(self.calc_flux(self.waref, self.trref, waz, flz))

        norm = self.refband_flambda / sedflux
        flz *= norm
        self.fl0 *= norm

        self.waz = waz
        self.flz = flz

    def passband(self, filepath):
        """
        Load a passband file and return sorted wavelength and transmission arrays.

        Args:
            filepath (str): Path to passband file.

        Returns:
            tuple: (wavelength array, transmission array)
        """
        wapass, trpass = np.loadtxt(filepath, usecols=(0, 1), unpack=True)
        isort = wapass.argsort()
        wapass = wapass[isort]
        trpass = trpass[isort]

        return wapass, trpass

    def reference_band(self):
        """
        Compute reference band flux density.

        Returns:
            tuple: (reference magnitude, reference flux density)
        """
        # self.fnu_ref = self.flambda_to_fnu(self.effective_wa(self.waref, self.trref), self.refband_flambda)
        self.fnu_ref = 10 ** (-0.4 * (self.mag_ref - 8.9))
        return self.mag_ref, self.fnu_ref

    def mag2flux(self, ABmag, wapass, trpass):
        """
        Convert AB magnitude to flux density in a given passband.

        Args:
            ABmag (float or array): AB magnitude(s).
            wapass (ndarray): Passband wavelengths.
            trpass (ndarray): Passband transmission.

        Returns:
            float or array: Flux density in passband.
        """
        fnu = 10 ** (-(ABmag + 48.6) / 2.5)
        effective_wl = self.effective_wa(wapass, trpass)
        flambda = c / (effective_wl * 1e-8) ** 2 * fnu * 1e-8

        return flambda

    def effective_wa(self, wapass, trpass):
        """
        Compute the effective wavelength of a passband.

        Args:
            wapass (ndarray): Passband wavelengths.
            trpass (ndarray): Passband transmission.

        Returns:
            float: Effective wavelength.
        """
        a = np.trapz(trpass * wapass)
        b = np.trapz(trpass / wapass)
        effective_wa = np.sqrt(a / b)
        return effective_wa

    def sed_band(self, passpath):
        """
        Compute magnitude and flux density in a given passband for all SEDs.

        Args:
            passpath (str): Path to passband file.

        Returns:
            tuple: (magnitudes, flux densities)
        """
        wapass, trpass = self.passband(passpath)
        #flux_ = np.array(self.calc_flux(wapass, trpass, self.waz, self.flz))
        # print("flux_ = ", flux_)
        # print(len(flux_))
        mag_ = self.calc_mag(wapass, trpass, self.waz, self.flz)
        # fluxnu_ = self.flambda_to_fnu(self.effective_wa(wapass, trpass), flux_)
        fluxnu_ = 10 ** (-0.4 * (mag_ - 8.9))

        return mag_, fluxnu_

    def calc_flux(self, wapass, trpass, wa0, fl0):
        """
        Calculate the flux in a given passband for provided SEDs.

        Args:
            wapass (ndarray): Passband wavelengths.
            trpass (ndarray): Passband transmission.
            wa0 (ndarray): SED wavelengths.
            fl0 (ndarray): SED fluxes.

        Returns:
            float or array: Flux in passband.
        """
        if not isinstance(wa0[0], np.ndarray):
            wa0 = np.array([wa0])
            fl0 = np.array([fl0])

        sedflux = []
        for k in range(wa0.shape[0]):
            wa0k = wa0[k]
            fl0k = fl0[k]
            if wa0k[0] > wapass[0] or wa0k[-1] < wapass[-1]:
                msg = "SED does not cover the whole bandpass, extrapolating"
                warnings.warn(msg)
                dw = np.median(np.diff(wa0k))
                sedwa = np.arange(wapass[0], wapass[-1] + dw, dw)
                sedfl = np.interp(sedwa, wa0k, fl0k)
            else:
                sedwa = wa0k
                sedfl = fl0k

            i, j = sedwa.searchsorted([wapass[0], wapass[-1]])
            fl = sedfl[i:j]
            wa = sedwa[i:j]

            dw_band = np.median(np.diff(wapass))
            dw_sed = np.median(np.diff(wa))
            if dw_sed > dw_band and dw_band > 20:
                warnings.warn(
                    'WARNING: SED wavelength sampling interval ~%.2f Ang, '
                    'but bandpass sampling interval ~%.2f Ang' %
                    (dw_sed, dw_band))
                fl = np.interp(wapass, wa, fl)
                band_tr = trpass
                wa = wapass
            else:
                band_tr = np.interp(wa, wapass, trpass)

            sedfluxi = np.trapz(band_tr * fl * wa, wa) / np.trapz(band_tr * wa, wa)
            sedflux.append(sedfluxi)

        return sedflux if len(sedflux) > 1 else sedflux[0]

    def flambda_to_fnu(self, wa, f_lambda):
        """
        Convert flux density from f_lambda to f_nu.

        Args:
            wa (float or array): Wavelength(s).
            f_lambda (float or array): Flux density per unit wavelength.

        Returns:
            float or array: Flux density per unit frequency.
        """
        return (wa * 1e-8) ** 2 * f_lambda * 1e8 / c

    def fnu_to_flambda(self, wa, f_nu):
        """
        Convert flux density from f_nu to f_lambda.

        Args:
            wa (float or array): Wavelength(s).
            f_nu (float or array): Flux density per unit frequency.

        Returns:
            float or array: Flux density per unit wavelength.
        """
        return c / (wa * 1e-8) ** 2 * f_nu * 1e-8

    def calc_mag(self, wapass, trpass, wased, flsed, system="AB"):
        """
        Calculate magnitudes in a given passband for provided SEDs.

        Args:
            wapass (ndarray): Passband wavelengths.
            trpass (ndarray): Passband transmission.
            wased (ndarray): SED wavelengths.
            flsed (ndarray): SED fluxes.
            system (str): Magnitude system (default "AB").

        Returns:
            ndarray: Magnitudes in passband.
        """
        # AB SED has constant flux density (f_nu) 3631 Jy, see
        # http://www.sdss.org/dr5/algorithms/fluxcal.html
        AB_fnu = 3631 * Jy  # erg/s/cm^2/Hz
        AB_wa = np.logspace(1, 10, 100000)  # Ang
        AB_flambda = self.fnu_to_flambda(AB_wa, AB_fnu)
        f1 = np.array(self.calc_flux(wapass, trpass, wased, flsed))

        mag = np.where(f1 > 0, -2.5 * np.log10(f1 / self.calc_flux(wapass, trpass, AB_wa, AB_flambda)), -np.inf)

        return mag
