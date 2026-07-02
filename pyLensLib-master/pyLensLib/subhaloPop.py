import numpy as np
from scipy import stats
from scipy import optimize


class subhaloPop(object):
    """
    Subhalo population generator and property calculator for lensing simulations.

    Attributes:
        zhalo (float): Host halo redshift.
        mmin, mA, mslope, md, mnorm (float): SHMF parameters.
        ralpha, rbeta, cvir (float): Radial distribution parameters.
        halo_ell2d, halo_pa (float): Host halo shape/orientation.
        sh_msigma_m0, sh_msigma_sigma0, sh_msigma_slope (float): Mass-velocity relation.
        sh_rcutsigma_sigma0, sh_rcutsigma_rcut0, sh_rcutsigma_slope (float): Velocity-radius relation.
        sh_sigmaL_M0, sh_sigmaL_slope (float): Velocity-luminosity relation.
        mr, r, pa, x, y, sigma, rcut, Mag, re: Subhalo properties.
        ngen (int): Number of generated subhalos.
    """

    def __init__(self,**kwargs):
        """
        Initialize subhalo population and calculate subhalo properties.

        Args:
            **kwargs: Subhalo population and property parameters.

        Returns:
            None
        """
        npts=100

        # parameters defining the SHMF
        if ('zhalo' in kwargs):
            self.zhalo = kwargs['zhalo']
        else:
            self.zhalo = 0.0

        if ('mmin' in kwargs):
            self.mmin = kwargs['mmin']
        else:
            self.mmin = 1e10

        if ('mA' in kwargs):
            self.mA = kwargs['mA']
        else:
            self.mA = -3.03

        if ('mslope' in kwargs):
            self.mslope = kwargs['mslope']
        else:
            self.mslope = -0.9

        if ('md' in kwargs):
            self.md = kwargs['md']
        else:
            self.md = 12.2715

        if ('mnorm' in kwargs):
            self.mnorm = kwargs['mnorm']
        else:
            self.mnorm = 1e15

        # parameters defining the radial distribution function
        if ('ralpha' in kwargs):
            self.ralpha = kwargs['ralpha']
        else:
            self.ralpha = 1.944

        if ('rbeta' in kwargs):
            self.rbeta = kwargs['rbeta']
        else:
            self.rbeta = 2.75

        if ('cvir' in kwargs):
            self.cvir = kwargs['cvir']
        else:
            self.cvir = 5.0

        # parameters defining the shape and orientation of the host halo
        if ('halo_ell2d' in kwargs):
            self.halo_ell2d = kwargs['halo_ell2d']
        else:
            self.halo_ell2d = 0.0

        if ('halo_pa' in kwargs):
            self.halo_pa = kwargs['halo_pa']
        else:
            self.halo_pa = 0.0

        # conversion of mass into velocity dispersion
        if ('sh_msigma_m0' in kwargs):
            self.sh_msigma_m0 = kwargs['sh_msigma_m0']
        else:
            self.sh_msigma_m0 = 3.5e11

        if ('sh_msigma_sigma0' in kwargs):
            self.sh_msigma_sigma0 = kwargs['sh_msigma_sigma0']
        else:
            self.sh_msigma_sigma0 = 220.0

        if ('sh_msigma_slope' in kwargs):
            self.sh_msigma_slope = kwargs['sh_msigma_slope']
        else:
            self.sh_msigma_slope = 4.43

        # conversion of velocity dispersion into r_cut (for PIEMD-like subhalos)
        if ('sh_rcutsigma_sigma0' in kwargs):
            self.sh_rcutsigma_sigma0 = kwargs['sh_rcutsigma_sigma0']
        else:
            self.sh_rcutsigma_sigma0 = 220.0

        if ('sh_rcutsigma_rcut0' in kwargs):
            self.sh_rcutsigma_rcut0 = kwargs['sh_rcutsigma_rcut0']
        else:
            self.sh_rcutsigma_rcut0 = 10.1

        if ('sh_rcutsigma_slope' in kwargs):
            self.sh_rcutsigma_slope = kwargs['sh_rcutsigma_slope']
        else:
            self.sh_rcutsigma_slope = 2.43

        if ('sh_sigmaL_M0' in kwargs):
            self.sh_sigmaL_M0 = kwargs['sh_sigmaL_M0']
        else:
            self.sh_sigmaL_M0 = -23.0

        if ('sh_sigmaL_slope' in kwargs):
            self.sh_sigmaL_slope = kwargs['sh_sigmaL_slope']
        else:
            self.sh_sigmaL_slope = 0.27


        xmin = np.log10(self.mmin/self.mnorm)

        x = np.logspace(xmin, 0, npts)
        dlogx = np.log10(x[1]) - np.log10(x[0])
        blog = []
        for i in range(len(x)):
            blog.append(10 ** (np.log10(x[i]) - dlogx / 2.0))
        blog.append(10 ** (np.log10(x[-1]) + dlogx / 2.0))
        mf = self.massf(x)
        self.ngen = int((np.array([mf[i] *
                                      (np.log10(blog[i + 1]) -
                                       np.log10(blog[i])) for i in range(len(mf))]).sum()))

        mf = np.array(mf).astype(float) / np.sum(mf)
        custm = stats.rv_discrete(name='custm', values=(range(len(x)), mf))
        R = custm.rvs(size=self.ngen)
        R_ = R + 1
        rnum = np.random.random_sample(self.ngen)
        self.mr = [10 ** ((np.log10(blog[R_[i]]) - np.log10(blog[R[i]])) *
                          rnum[i] + np.log10(blog[R[i]])) for i in range(self.ngen)]
        self.mr = np.sort(np.array(self.mr) * self.mnorm)[::-1]

        rn_ = np.random.random_sample(self.ngen-1)
        r = [0.0]
        for rn in rn_:
            r.append(self.generate_radius(rn))

        self.r = np.array(r)

        # assign a position angle to each subhalo
        self.pa = np.random.random_sample(self.ngen)*np.pi*2.0
        # add ellipticity
        self.addEllipticity()

        # central velocity dispersion and truncation radius
        self.sigma_from_mass()
        self.rcut_from_sigma()
        self.luminisity_from_sigma()

    def massf(self,x):
        """
        Subhalo mass function (SHMF).

        Args:
            x (array): Mass array (normalized).

        Returns:
            array: SHMF values for each mass.
        """
        A = np.log(10)*10**self.mA
        zresc = (1.0 + self.zhalo) ** 0.5
        return zresc * A * (x*self.mnorm)**self.mslope*np.exp(-self.md*x**3)*self.mnorm

    def raddist(self,x):
        """
        Radial distribution function for subhalos.

        Args:
            x (array): Radial positions (normalized).

        Returns:
            array: Radial distribution values.
        """
        return 1.0 - (1 + self.ralpha * self.cvir) * x ** self.rbeta / (1 + self.ralpha * self.cvir * x ** 2)

    def generate_radius(self, rn):
        """
        Generate a subhalo radius from the radial distribution function.

        Args:
            rn (float): Random number in [0, 1].

        Returns:
            float: Generated radius (normalized).
        """

        def raddist_cdf(x):
            return 1.0 - (1 + self.ralpha * self.cvir) * x ** self.rbeta / (1 + self.ralpha * self.cvir * x ** 2) - rn

        root = optimize.brentq(raddist_cdf, 0, 1)
        return root

    def addEllipticity(self):
        """
        Add ellipticity to subhalo positions based on host halo shape and orientation.

        Returns:
            None
        """
        x=self.r*np.cos(self.pa)
        y=self.r*np.sin(self.pa)
        old = np.array(list(zip(x,y)))
        ellipt_matrix = np.array([[1.0 + self.halo_ell2d * np.cos(2.0 * self.halo_pa),
                                   self.halo_ell2d * np.sin(2.0 * self.halo_pa)],
                                  [self.halo_ell2d * np.sin(2.0 * self.halo_pa),
                                   1.0 - self.halo_ell2d * np.cos(2.0 * self.halo_pa)]])
        new = np.dot(ellipt_matrix, old.T)
        self.x, self.y = new[0, :], new[1, :]

    def sigma_from_mass(self):
        """
        Calculate central velocity dispersion for each subhalo from its mass.

        Returns:
            None
        """
        self.sigma=self.sh_msigma_sigma0*(self.mr/self.sh_msigma_m0)**(1.0/self.sh_msigma_slope)

    def rcut_from_sigma(self):
        """
        Calculate truncation radius for each subhalo from its velocity dispersion.

        Returns:
            None
        """
        self.rcut=self.sh_rcutsigma_rcut0*(self.sigma/self.sh_rcutsigma_sigma0)**self.sh_rcutsigma_slope

    def rcut_from_sigma(self):
        self.rcut=self.sh_rcutsigma_rcut0*(self.sigma/self.sh_rcutsigma_sigma0)**self.sh_rcutsigma_slope

    def luminisity_from_sigma(self):
        """
        Calculate luminosity in the K-band from velocity dispersion.

        Returns:
            None
        """
        """
        # \sigma=\sigma0*(L/L0)^\alpha meaning that
        # L/L0=(\sigma/\sigma0)^1/\alpha
        # in Mag: M-M0=-2.5log10(L/L0)=-2.5log10(\sigma/\sigma0)^1/\alpha)
        # assume M0=-23 (Bergamini et al. 2019)
        :return:
        Luminosity in the K-band
        """
        self.Mag=self.sh_sigmaL_M0 - 2.5*np.log10((self.sigma/self.sh_msigma_sigma0)**(1.0/self.sh_sigmaL_slope))

    def re_from_rcut(self):
        """
        Calculate effective radius from truncation radius for each subhalo.

        Returns:
            None
        """
        self.re = self.rcut*3.0/4.0