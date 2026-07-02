import numpy as np
import pyLensLib.sedcompat as ss
import scipy.interpolate as interpolate
from matplotlib import pyplot as plt
from pathlib import Path

#udfdir='/Users/massimo/stiva/HUDF/'
#homedir='/Users/massimo/CODES/bpz-1.99.3/SED/'

ref=''
filtern0=''


class srccatalog(object):
    """
    Source catalog for lensing simulations, including photometric and SED data.

    Attributes:
        df (pandas.DataFrame): Catalog data.
        fov (float): Field of view in arcseconds.
        filtern (str): Filter response file.
        xdfband (str): Reference band.
        recal (str): Recalibration flag.
        mlim (float): Magnitude limit.
        seed (int): Random seed.
        udfdir (str): Directory for HUDF data.
        homedir (str): Directory for SED templates.
    """

    def __init__(self,filename=None,**kwargs):
        """
        Initialize the source catalog and load photometric/SED data.

        Args:
            filename (str, optional): Output filename for catalog.
            **kwargs: Catalog and simulation parameters.

        Returns:
            None
        """
        if ('FOV' in kwargs):
            self.fov = kwargs['FOV']
        else:
            self.fov = 200.0

        if ('filter' in kwargs):
            self.filtern = kwargs['filter']
        else:
            self.filtern = 'BPZ/HST_ACS_WFC_F606W.res'

        if ('usedband' in kwargs):
            self.xdfband = kwargs['useband']
        else:
            self.xdfband = 'f775w'

        if ('recal' in kwargs):
            self.recal = kwargs['recal']
        else:
            self.recal = 'yes'

        if ('maglim' in kwargs):
            self.mlim = kwargs['maglim']
        else:
            self.mlim = 28.0

        if ('seed' in kwargs):
            self.seed = kwargs['seed']
        else:
            self.seed = 123

        if ('udfdir' in kwargs):
            self.udfdir = kwargs['udfdir']
        else:
            self.udfdir = "/Users/massimo/stiva/HUDF/"

        if ('homedir' in kwargs):
            self.homedir = kwargs['homedir']
        else:
            self.homedir = "/Users/massimo/CODES/bpz-1.99.3/SED/"

        id, coe_id, f225w, f275w, f336w, f435w, f606w, f775w, f850lp, f105w, f125w, f140w, f160w, z, st = np.loadtxt(
            self.udfdir + 'uvudf_rafelski_2015.dat', unpack=True,
            usecols=[0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 80, 85])

        sed_list = 'eB11.list'

        ninterp = 9
        ref, filtern0 = self.refband(self.xdfband,
                                     f225w, f275w, f336w, f435w,
                                     f606w, f775w, f850lp, f105w,
                                     f125w, f140w, f160w)

        if (self.recal == 'yes'):
            # get SED templates
            sedl = self.readList(sed_list)
            nprimary = sedl.size
            # spectral types to evaluate
            st_ev = np.linspace(1.0, float(nprimary), (nprimary - 1) * ninterp + nprimary)
            nofseds = st_ev.size
            # generate the sed objects
            l, f, = self.readSed(sedl[0])
            sed = [ss.SED(wa=l, fl=f, z=0)]
            for i in range(nofseds - 1):
                if i > 0:
                    l, f, = self.intermediateSed(st_ev[i], sedl, ninterp)
                    sed.append(ss.SED(wa=l, fl=f, z=0))
            l, f = self.readSed(sedl[sedl.size - 1])
            sed.append(ss.SED(wa=l, fl=f, z=0))
            passband = ss.Passband(self._resolve_filter_path(self.filtern))
            passband0 = ss.Passband(self._resolve_filter_path(filtern0))
            mnew = np.zeros(id.size)
            for i in range(id.size):
                ised = self.sedNumber(st[i], ninterp)
                mnorm = ref[i]
                sed[ised].normalise_to_mag(mnorm, passband0)
                mnew[i] = sed[ised].calc_mag(passband, system='AB')
        else:
            mnew=ref

        area_udf = 10.8  # sq. arcmin
        area = (self.fov / 60.0) * (self.fov / 60.0)
        ngal = int(np.rint(id.size * area / area_udf))

        npp = np.random.RandomState(self.seed)

        ind = npp.randint(0, id.size, ngal)
        xgal = (npp.random_sample(ind.size) - 0.5) * self.fov
        ygal = (npp.random_sample(ind.size) - 0.5) * self.fov
        pagal = npp.random_sample(ind.size) * np.pi * 2.0
        fx = npp.random_sample(ind.size)
        fy = npp.random_sample(ind.size)
        flipx = np.zeros(ind.size)
        flipy = np.zeros(ind.size)
        for i in range(ind.size):
            flipx[i] = int(np.round(fx[i]))
            flipy[i] = int(np.round(fy[i]))

        zgal = np.zeros(ind.size)

        mgal = np.zeros(ind.size)
        stgal = np.zeros(ind.size)
        idgal = np.zeros(ind.size)
        idcgal = np.zeros(ind.size)

        for i in range(ind.size):
            zgal[i] = z[ind[i]]
            mgal[i] = mnew[ind[i]]
            stgal[i] = st[ind[i]]
            idgal[i] = id[ind[i]]
            idcgal[i] = coe_id[ind[i]]

        sm = mgal < self.mlim
        # create dictionary to be stored in a pandas dataframe
        res={
            'idgal': idgal[sm],
            'idcgal': idcgal[sm],
            'zgal': zgal[sm],
            'mag': mgal[sm],
            'template': stgal[sm],
            'x': xgal[sm],
            'y': ygal[sm],
            'PA': pagal[sm],
            'flipx': flipx[sm],
            'flipy': flipy[sm]
        }
        import pandas as pd
        self.df = pd.DataFrame(res)
        self.df['idgal'] = self.df['idgal'].astype(int)
        self.df['idcgal'] = self.df['idcgal'].astype(int)
        self.df['flipx'] = self.df['flipx'].astype(int)
        self.df['flipy'] = self.df['flipy'].astype(int)

        if filename != None:
            self.df.to_csv(filename,sep=' ',index=False)
            # Do not return anything in __init__
        # Do not return self.df in __init__
        #arr = list(zip(idgal[sm], idcgal[sm], zgal[sm], mgal[sm], stgal[sm], xgal[sm], ygal[sm], pagal[sm], flipx[sm],
        #          flipy[sm]))
        #np.savetxt(filename, arr, fmt='%i %i %f %f %f %f %f %f %i %i')

    def _resolve_filter_path(self, filter_name):
        """
        Resolve BPZ-style filter names to real files.

        Typical inputs are strings like:
        - 'BPZ/HST_ACS_WFC_F606W.res'
        - 'HST_ACS_WFC_F606W.res'
        - absolute filesystem paths
        """
        p = Path(str(filter_name)).expanduser()
        if p.is_file():
            return p.as_posix()

        rel = str(filter_name)
        if rel.startswith("BPZ/"):
            rel = rel.split("/", 1)[1]

        sed_dir = Path(self.homedir).expanduser()
        if sed_dir.name.lower() == "sed":
            bpz_root = sed_dir.parent
        else:
            bpz_root = sed_dir

        candidates = [
            bpz_root / rel,
            bpz_root / Path(rel).name,
            bpz_root / "FILTER" / rel,
            bpz_root / "FILTER" / Path(rel).name,
            sed_dir / rel,
            sed_dir / Path(rel).name,
        ]
        for c in candidates:
            if c.is_file():
                return c.as_posix()

        raise FileNotFoundError(
            f"Filter file '{filter_name}' not found. "
            f"Tried relative to homedir='{self.homedir}' and BPZ/FILTER conventions."
        )

    def readList(self,filename):
        """
        Read a list of SED template filenames.

        Args:
            filename (str): Filename of the SED list.

        Returns:
            ndarray: Array of SED filenames.
        """
        sedf = np.loadtxt(self.homedir + filename, unpack=True, dtype=str, usecols=[0])
        return (sedf)

    def readSed(self,filename):
        """
        Read and normalize a SED template.

        Args:
            filename (str): SED template filename.

        Returns:
            tuple: Wavelength and normalized flux arrays.
        """
        spectra_file = self.homedir + filename
        lo, fo = np.loadtxt(spectra_file, unpack=True, dtype=float, usecols=[0, 1])
        fi = interpolate.interp1d(lo, fo, 'linear')
        fn = fi(15000.0)
        # fo=fo/fn
        l = np.linspace(1000.0, 25000.0, 3000)
        f = fi(l) / fn
        return (l, f)

    def refband(self,xdfband,f225w, f275w, f336w, f435w,f606w, f775w, f850lp, f105w,f125w, f140w, f160w):
        """
        Select the reference band and filter for a given band name.

        Args:
            xdfband (str): Band name.
            f225w ... f160w (array): Magnitudes in each band.

        Returns:
            tuple: Reference magnitude array and filter filename.
        """
        global ref, filtern0
        if xdfband == 'f225w':
            ref = f225w
            filtern0 = 'BPZ/HST_WFC3_UVIS_F225W.res'
        if xdfband == 'f275w':
            ref = f275w
            filtern0 = 'BPZ/HST_WFC3_UVIS_F275W.res'
        if xdfband == 'f336w':
            ref = f336w
            filtern0 = 'BPZ/HST_WFC3_UVIS_F336W.res'
        if xdfband == 'f435w':
            ref = f435w
            filtern0 = 'BPZ/HST_ACS_WFC_F435W.res'
        if xdfband == 'f606w':
            ref = f606w
            filtern0 = 'BPZ/HST_ACS_WFC_F606W.res'
        if xdfband == 'f775w':
            ref = f775w
            filtern0 = 'BPZ/HST_ACS_WFC_F775W.res'
        if xdfband == 'f850lp':
            ref = f850lp
            filtern0 = 'BPZ/HST_ACS_WFC_F850LP.res'
        if xdfband == 'f105w':
            ref = f105w
            filtern0 = 'BPZ/HST_WFC3_IR_F105W.res'
        if xdfband == 'f125w':
            ref = f125w
            filtern0 = 'BPZ/HST_WFC3_IR_F125W.res'
        if xdfband == 'f140w':
            ref = f140w
            filtern0 = 'BPZ/HST_WFC3_IR_F140W.res'
        if xdfband == 'f160w':
            ref = f160w
            filtern0 = 'BPZ/HST_WFC3_IR_F160W.res'
        return (ref, filtern0)

    def specT2primarySed(self,s):
        """
        Convert spectral type to primary SED index.

        Args:
            s (float): Spectral type.

        Returns:
            int: Primary SED index.
        """
        x = int(s) - 1
        # print 'primary:',s,x
        return (x)

    def specT2secondarySed(self,s, x, n):
        """
        Convert spectral type to secondary SED index.

        Args:
            s (float): Spectral type.
            x (int): Primary SED index.
            n (int): Number of interpolations.

        Returns:
            int: Secondary SED index.
        """
        # n=INTERP
        y = int(np.rint((s - x - 1.0) * (n + 1)))
        # print 'secondary:', s,x,s-x,y
        return (y)

    def sedNumber(self,s, n):
        """
        Get the SED number for a given spectral type and interpolation.

        Args:
            s (float): Spectral type.
            n (int): Number of interpolations.

        Returns:
            int: SED number.
        """
        x = self.specT2primarySed(s)
        y = self.specT2secondarySed(s, x, n)
        # print s,x,y
        nsed = int(np.rint(x * (n + 1) + y))
        return (nsed)

    def intermediateSed(self,st, sedl, ninterp):
        """
        Generate an intermediate SED by interpolating between templates.

        Args:
            st (float): Spectral type.
            sedl (array): List of SED filenames.
            ninterp (int): Number of interpolations.

        Returns:
            tuple: Wavelength and interpolated flux arrays.
        """
        psed0 = self.specT2primarySed(st)
        psed1 = psed0 + 1
        l0, f0 = self.readSed(sedl[psed0])
        l1, f1 = self.readSed(sedl[psed1])
        l = l0
        ised = self.specT2secondarySed(st, psed0, ninterp)
        f = f0 + (f1 - f0) / (ninterp + 1) * ised
        return (l, f)

    def get_dataframe(self):
        """
        Return the pandas DataFrame containing the catalog data.

        Returns:
            pandas.DataFrame: Catalog data.
        """
        return self.df

    def dlensing(self, cosmo, z_lens, z_source):
        """
        Compute the lensing distance ratio D_L * D_LS / D_S.

        Args:
            cosmo: Astropy cosmology object.
            z_lens (float): Lens redshift.
            z_source (float): Source redshift.

        Returns:
            float: Lensing distance ratio in Mpc.
        """
        # Compute distances
        D_L = cosmo.angular_diameter_distance(z_lens).value  # in Mpc
        D_S = cosmo.angular_diameter_distance(z_source).value  # in Mpc
        D_LS = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value  # in Mpc

        # Compute D_L * D_LS / D_S
        dl_dls_ds_ratio = D_L * D_LS / D_S
        return dl_dls_ds_ratio

    # generate n_planes values of dl
    def generateSrcPlanesDL(self, cosmo, z_lens, z_source_max, n_planes=100):
        """
        Generate source plane redshifts and lensing distances for multiple planes.

        Args:
            cosmo: Astropy cosmology object.
            z_lens (float): Lens redshift.
            z_source_max (float): Maximum source redshift.
            n_planes (int, optional): Number of planes.

        Returns:
            tuple: Arrays of source plane redshifts and lensing distances.
        """
        dl_max = self.dlensing(cosmo, z_lens, z_source_max)
        zs_arr = np.linspace(z_lens, z_source_max, 1000)
        dl_dls_ds = self.dlensing(cosmo, z_lens, zs_arr)

        interp_func = interpolate.interp1d(dl_dls_ds, zs_arr, bounds_error=False, fill_value=np.nan)
        dl_arr_equisp = np.linspace(0, dl_max, n_planes)
        z_planes = interp_func(dl_arr_equisp)
        z_planes[0] = 0.0
        return z_planes, dl_arr_equisp

    if __name__ == "__main__":
            # Example usage
            from pyLensLib.srccatalog import srccatalog
            from astropy.cosmology import FlatLambdaCDM
            kwargs_src = {
                'FOV': 100.0,
                'filtern': 'BPZ/HST_ACS_WFC_F606W.res',
                'useband': 'f775w',
                'recal': 'yes',
                'maglim': 30.0,
                'udfdir': '/Users/maxmen3/stiva/HUDF/',
                'homedir': '/Users/maxmen3/CODES/bpz-1.99.3/SED/'
            }
            catalog = srccatalog(**kwargs_src)
            df = catalog.get_dataframe()

            cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
            zlens = 0.4
            z_planes, dl_arr_equisp = catalog.generateSrcPlanesDL(cosmo, z_lens=zlens, z_source_max=10.0, n_planes=100)
            #plt.plot(z_planes, dl_arr_equisp)
            #plt.xlabel('Source redshift (z)')
            #plt.ylabel('D_L * D_LS / D_S (Mpc)')
            #plt.title('Source Plane Distances')
            #plt.grid()
            #plt.show()
            # place sources on the planes
            z_source = df['zgal'].values

            plane_indices = np.zeros(len(z_source), dtype=int)
            for i in range(len(z_source)):
                if z_source[i] <= zlens:
                    plane_indices[i] = 0
                else:
                    plane_indices[i] = np.abs(z_source[i] - z_planes).argmin()

            df['plane_index'] = plane_indices
            df['plane_z'] = z_planes[plane_indices]

            print(df.head(50))

