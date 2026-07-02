import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.io.fits as pyfits
from scipy.ndimage import map_coordinates

class lensplane(object):
    """
    Represents a single lens plane for multiplane lensing.

    Args:
        inputdata (str or list): FITS filename or list of lens objects.
        fromfits (bool, optional): If True, read from FITS file. If False, build from lens list.
    """
    def __init__(self, inputdata, fromfits=True):
        """
        Initialize a lensplane object from FITS file or lens list.

        Args:
            inputdata (str or list): FITS filename or list of lens objects.
            fromfits (bool, optional): If True, read from FITS file. If False, build from lens list.
        """
        if fromfits:
            # read the deflection angle maps from the fits file
            myhd = pyfits.open(inputdata)
            self.a1 = myhd[0].data
            self.a2 = myhd[1].data
            # read lens plane parameters from header of the fits file:
            self.z = myhd[0].header['ZLENS']
            self.xmin = myhd[0].header['XMIN']
            self.xmax = myhd[0].header['XMAX']
            self.ymin = myhd[0].header['YMIN']
            self.ymax = myhd[0].header['YMAX']
            omega = myhd[0].header['OMEGA']
            hubble = myhd[0].header['H']
            self.co = FlatLambdaCDM(H0=hubble * 100.0, Om0=omega)
            # compute the pixel scale:
            self.px = (self.xmax - self.xmin) / (myhd[0].header['NAXIS1'] - 1)
            try:
                self.zs = myhd[0].header['ZS']
            except KeyError:
                self.zs = 0.0
            if (self.zs > 0):
                # if a source redshift is assumed, the deflection angles are reduced.
                # Then, remove the D_LS/D_S factor to obtain the absolute deflections:
                dls = self.co.angular_diameter_distance_z1z2(self.z, self.zs)
                ds = self.co.angular_diameter_distance(self.zs)
                self.a1 = self.a1 * ds / dls
                self.a2 = self.a2 * ds / dls
        else:
            # create a lens plane from a list of lenses
            nlenses = len(inputdata)
            ilens = 0
            for gl in inputdata:
                try:
                    _ = gl.theta1.shape
                except NameError:
                    print (("Lens %i of %i does not have a grid yet. Please use setGrid to initialize it") % (ilens,nlenses))
                if ilens>0:
                    gl_.combinewith(gl)
                    if gl.zs != zscurr or gl.zl != zlcurr:
                        raise ValueError("All lenses on the same lens plane should share the same zl and zs")
                else:
                    gl_=gl
                    zscurr = gl.zs
                    zlcurr = gl.zl
                ilens+=1
            self.a1 = gl_.a1
            self.a2 = gl_.a2
            self.z = gl_.zl
            self.zs = gl_.zs
            self.xmin = gl_.theta1.min()
            self.ymin = self.xmin
            self.xmax = gl_.theta1.max()
            self.ymax = self.xmax
            self.co = gl_.co
            # compute the pixel scale:
            self.px = (self.xmax - self.xmin) / (len(gl_.theta1) - 1)
            dls = self.co.angular_diameter_distance_z1z2(self.z, self.zs)
            ds = self.co.angular_diameter_distance(self.zs)
            self.a1 = self.a1 * ds / dls
            self.a2 = self.a2 * ds / dls

    def angles(self, theta1, theta2):
        """
        Interpolate the deflection angle components at given coordinates.

        Args:
            theta1 (float or array): X coordinate(s).
            theta2 (float or array): Y coordinate(s).

        Returns:
            tuple: (a1, a2) interpolated deflection angles.
        """
        theta1pix = (theta1 - self.xmin) / self.px
        theta2pix = (theta2 - self.ymin) / self.px
        a1 = map_coordinates(self.a1, [theta2pix, theta1pix], order=2)
        a2 = map_coordinates(self.a2, [theta2pix, theta1pix], order=2)
        return (a1, a2)

class multiplane(object):

    def __init__(self, lp, zs=2.0):
        """
        Initialize a multiplane deflector using a list of lensplanes and a source redshift.

        Args:
            lp (list): List of lensplane objects.
            zs (float, optional): Source redshift. Default is 2.0.
        """
        if isinstance(lp, list):
            if len(lp) == 0:
                raise ValueError("lp must contain at least one lensplane.")
            lp_input = lp
        else:
            lp_input = [lp]

        self.co = lp_input[0].co
        # Keep only foreground planes and sort by redshift so distance steps are physical.
        self.lp = sorted([plane for plane in lp_input if plane.z < zs], key=lambda plane: plane.z)
        zl_ = [plane.z for plane in self.lp]
        self.nlens = len(self.lp)
        # append the source plane redshift
        zl_.append(zs)

        # loop over the lens planes to initialize the distances
        # differences between transverse distances on planes i,j
        self.deltaT_list = []
        # distances between the observer and lens planes
        self.T_list = []

        # start from redshift z=0
        z_before = 0.0
        for z_lens in zl_[:-1]:
            # calculate the differences between the transverse distances in
            # consecutive pairs of lens planes
            T_last = self.co.comoving_transverse_distance(z_lens)
            T_before = self.co.comoving_transverse_distance(z_before)
            delta_T = T_last - T_before
            self.deltaT_list.append(delta_T.value)
            # save transverse comoving distance for lens plane
            self.T_list.append(T_last.value)
            z_before = z_lens
        # add deltaT between source plane and last lens plane
        T_source = self.co.comoving_transverse_distance(zs)
        T_before = self.co.comoving_transverse_distance(z_before)
        delta_T = T_source - T_before
        self.deltaT_list.append(delta_T.value)
        # save the comoving transverse distance at the source redshift
        self.T_source = T_source.value

    def com2rad_source(self, x_1, x_2):
        """
        Compute angular positions of light rays on the source plane.

        Args:
            x_1 (float or array): X coordinate(s).
            x_2 (float or array): Y coordinate(s).

        Returns:
            tuple: (theta_1, theta_2) angular positions.
        """
        T = self.T_source
        theta_1 = x_1 / T
        theta_2 = x_2 / T
        return theta_1, theta_2

    def com2rad(self, x_1, x_2, idex):
        """
        Compute angular positions of light rays on a lens plane.

        Args:
            x_1 (float or array): X coordinate(s).
            x_2 (float or array): Y coordinate(s).
            idex (int): Lens plane index.

        Returns:
            tuple: (theta_1, theta_2) angular positions.
        """
        T = self.T_list[idex]
        theta_1 = x_1 / T
        theta_2 = x_2 / T
        return theta_1, theta_2

    def raytrace(self, theta_1, theta_2):
        """
        Propagate the light rays from the first lens plane towards the sources.

        Args:
            theta_1 (float or array): Initial X angular position(s).
            theta_2 (float or array): Initial Y angular position(s).

        Returns:
            tuple: (beta_1, beta_2) angular positions on the source plane.
        """
        x1 = np.zeros_like(theta_1)
        x2 = np.zeros_like(theta_2)
        alpha_1 = theta_1
        alpha_2 = theta_2
        for i in range(self.nlens):
            delta_T = self.deltaT_list[i]
            x1, x2 = self.next_step(x1, x2, alpha_1, alpha_2, delta_T)
            alpha_1, alpha_2 = self.Talpha(x1, x2, alpha_1, alpha_2, i)
        delta_T = self.deltaT_list[self.nlens]
        x1, x2 = self.next_step(x1, x2, alpha_1, alpha_2, delta_T)
        beta_1, beta_2 = self.com2rad_source(x1, x2)
        return beta_1, beta_2

    def next_step(self, x1, x2, alpha_1, alpha_2, delta_T):
        """
        Compute arrival position of light ray on current plane.

        Args:
            x1 (float or array): Previous X position(s).
            x2 (float or array): Previous Y position(s).
            alpha_1 (float or array): Deflection X component(s).
            alpha_2 (float or array): Deflection Y component(s).
            delta_T (float): Transverse distance increment.

        Returns:
            tuple: (x1_, x2_) new positions.
        """
        x1_ = x1 + alpha_1 * delta_T
        x2_ = x2 + alpha_2 * delta_T

        return x1_, x2_

    def Talpha(self, x1, x2, alpha_1, alpha_2, idex):
        """
        Compute Talpha at step idex.

        Args:
            x1 (float or array): X position(s).
            x2 (float or array): Y position(s).
            alpha_1 (float or array): Deflection X component(s).
            alpha_2 (float or array): Deflection Y component(s).
            idex (int): Lens plane index.

        Returns:
            tuple: (Talpha_1, Talpha_2) updated deflection components.
        """
        theta_1, theta_2 = self.com2rad(x1, x2, idex)
        alpha_1_arcsec, alpha_2_arcsec = \
            self.lp[idex].angles(np.rad2deg(theta_1) * 3600.0,
                                 np.rad2deg(theta_2) * 3600.0)
        Talpha_1 = alpha_1 - np.deg2rad(alpha_1_arcsec / 3600.0)
        Talpha_2 = alpha_2 - np.deg2rad(alpha_2_arcsec / 3600.0)

        return Talpha_1, Talpha_2

    def alpha(self, theta1, theta2, beta1, beta2):
        """
        Compute the deflection angle between image and source positions.

        Args:
            theta1 (float or array): Image X position(s).
            theta2 (float or array): Image Y position(s).
            beta1 (float or array): Source X position(s).
            beta2 (float or array): Source Y position(s).

        Returns:
            tuple: (a1, a2) deflection angles.
        """
        a1, a2 = theta1 - beta1, theta2 - beta2
        return (a1, a2)
