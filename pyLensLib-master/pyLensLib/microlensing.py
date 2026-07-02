import numpy as np
from astropy import constants as const
from astropy import units as u

# this class deals with the source in the microlensing event
class point_source(object):
    """
    Represents a point source in a microlensing event.

    Args:
        flux (float, optional): Source flux. Default is 1.0.
        ds (float, optional): Source distance. Default is 10.0.
        vel (float, optional): Source velocity. Default is 200.
    """
    def __init__(self, flux=1.0, ds=10.0, vel=200.):
        self.ds = ds
        self.flux = flux
        self.vel = vel
        
# this class deals with the lens. It requires a point source to be 
# provided in order to build the point lens
class point_lens(object):
    """
    Represents a point lens in a microlensing event.

    Args:
        ps (point_source): Point source object.
        mass (float, optional): Lens mass (solar masses). Default is 1.0.
        dl (float, optional): Lens distance. Default is 5.0.
        ds (float, optional): Source distance. Default is 8.0.
        t0 (float, optional): Time of closest approach. Default is 0.0.
        y0 (float, optional): Minimum impact parameter. Default is 0.1.
    """
    def __init__(self, ps, mass=1.0, dl=5.0, ds=8.0, t0=0.0, y0=0.1):
        self.M = mass
        self.dl = dl
        self.ps = ps
        self.y0 = y0
        self.t0 = t0
        self.tE = self.EinsteinCrossTime()
    
    def EinsteinRadius(self):
        """
        Compute the Einstein radius for the lens.

        Returns:
            astropy.units.Quantity: Einstein radius in arcseconds.
        """
        mass=self.M*const.M_sun
        G=const.G
        c=c=const.c
        aconv=180.0*3600.0/np.pi*u.arcsecond
        return((np.sqrt(4.0*(G*mass/c/c).to('kpc')*(self.ps.ds-self.dl)
                        /self.dl/self.ps.ds/u.kpc))*aconv)
    
    def EinsteinCrossTime(self):
        """
        Compute the Einstein radius crossing time for the lens.

        Returns:
            astropy.units.Quantity: Einstein crossing time in days.
        """
        theta_e=self.EinsteinRadius()
        return(((theta_e.to('radian').value*self.dl*u.kpc).to('km')
                /self.ps.vel/u.km*u.s).to('day'))
    
    def y(self, t):
        """
        Compute the coordinates of the unlensed source at time t.

        Args:
            t (array-like): Time(s) at which to compute source position.

        Returns:
            tuple: (y1, y2) coordinates of the source.
        """
        y1=(t-self.t0)/self.tE.value
        y2=np.ones(len(t))*self.y0
        return(y1,y2)
    
    def xp(self, t):
        """
        Compute the coordinates of the x_+ image at time t.

        Args:
            t (array-like): Time(s) at which to compute image position.

        Returns:
            tuple: (xp1, xp2) coordinates of the x_+ image.
        """
        y1, y2  = self.y(t)
        Q = np.sqrt(y1**2 + y2**2 +4)/(np.sqrt(y1**2 + y2**2))
        xp1= 0.5 *(1 + Q)* y1
        xp2= 0.5 *(1 + Q)* y2
        return(xp1, xp2)
    
    def xm(self, t):
        """
        Compute the coordinates of the x_- image at time t.

        Args:
            t (array-like): Time(s) at which to compute image position.

        Returns:
            tuple: (xm1, xm2) coordinates of the x_- image.
        """
        y1, y2  = self.y(t)
        Q = np.sqrt(y1**2 + y2**2 +4)/(np.sqrt(y1**2 + y2**2))
        xm1= 0.5 *(1 - Q)* y1
        xm2= 0.5 *(1 - Q)* y2
        return(xm1, xm2)
    
    def mup(self, t):
        """
        Compute the magnification of the x_+ image at time t.

        Args:
            t (array-like): Time(s) at which to compute magnification.

        Returns:
            np.ndarray: Magnification of the x_+ image.
        """
        y1, y2  = self.y(t)
        yy=np.sqrt(y1**2+y2**2)
        mup=0.5*(1+(yy**2+2)/yy/np.sqrt(yy**2+4))
        return (mup)
    
    def mum(self, t):
        """
        Compute the magnification of the x_- image at time t.

        Args:
            t (array-like): Time(s) at which to compute magnification.

        Returns:
            np.ndarray: Magnification of the x_- image.
        """
        y1, y2  = self.y(t)
        yy=np.sqrt(y1**2+y2**2)
        mum=0.5*(1-(yy**2+2)/yy/np.sqrt(yy**2+4))
        return (mum)
    
    def xc(self, t):
        """
        Compute the coordinate of the light centroid at time t.

        Args:
            t (array-like): Time(s) at which to compute centroid.

        Returns:
            np.ndarray: Centroid coordinates.
        """
        xp=self.xp(t)
        xm=self.xm(t)
        xc=(xp*np.abs(self.mup(t))+xm*np.abs(self.mum(t)))/(np.abs(self.mup(t))+np.abs(self.mum(t)))
        return (xc)
    
    ################################################################################################
    def xp_ext_source(self, t, r):
        """
        Compute the coordinates of the x_+ image for an extended source at time t and radius r.

        Args:
            t (array-like): Time(s) at which to compute image position.
            r (float): Source radius.

        Returns:
            tuple: (xp1, xp2) coordinates for the extended source.
        """
        phi=np.linspace(0.0,2*np.pi,360)
        dy1=r*np.cos(phi)
        dy2=r*np.sin(phi)
        y1,y2=self.y(t)
        yy1=y1+dy1
        yy2=y2+dy2
        Q=np.sqrt(yy1**2+yy2**2+4.0)/np.sqrt(yy1**2+yy2**2)
        xp1=0.5*(1+Q)*yy1
        xp2=0.5*(1+Q)*yy2
        return(xp1,xp2)   
    
    def xm_ext_source(self, t, r):
        """
        Compute the coordinates of the x_- image for an extended source at time t and radius r.

        Args:
            t (array-like): Time(s) at which to compute image position.
            r (float): Source radius.

        Returns:
            tuple: (xm1, xm2) coordinates for the extended source.
        """
        phi=np.linspace(0.0,2*np.pi,360)
        dy1=r*np.cos(phi)
        dy2=r*np.sin(phi)
        y1,y2=self.y(t)
        yy1=y1+dy1
        yy2=y2+dy2
        Q=np.sqrt(yy1**2+yy2**2+4.0)/np.sqrt(yy1**2+yy2**2)
        xm1=0.5*(1-Q)*yy1
        xm2=0.5*(1-Q)*yy2
        return(xm1,xm2)
    
    def deltaxc(self, t):
        """
        Compute the offset of the centroid at time t.

        Args:
            t (array-like): Time(s) at which to compute centroid offset.

        Returns:
            tuple: (dx, dy) centroid offset.
        """
        y1,y2=self.y(t)
        yy=(y1**2+y2**2)
        return(y1/(yy+2),y2/(yy+2))


class binary_lens(object):
    """
    Represents a binary lens system for microlensing.

    Args:
        dl (float, optional): Lens distance. Default is 5.0.
        ds (float, optional): Source distance. Default is 8.0.
        m1 (float, optional): Mass of secondary lens. Default is 1.0.
        q (float, optional): Mass ratio m1/m2. Default is 1.0.
        d (float, optional): Separation between lenses (Einstein radius units). Default is 2.0.
        t0 (float, optional): Time of closest approach. Default is 0.0.
        y0 (float, optional): Minimum impact parameter. Default is 0.1.
        theta (float, optional): Inclination of source trajectory. Default is pi/4.
        centre_m2 (bool, optional): If True, t0/y0 refer to closest approach to primary lens (m2).
    """

    def __init__(self, dl=5.0, ds=8, m1=1.0, q=1.0, d=2.0, t0=0.0, y0=0.1, theta=np.pi / 4, centre_m2=False):
        """
        Initialize a binary lens object.

        Args:
            dl (float, optional): Lens distance.
            ds (float, optional): Source distance.
            m1 (float, optional): Mass of secondary lens.
            q (float, optional): Mass ratio m1/m2.
            d (float, optional): Separation between lenses (Einstein radius units).
            t0 (float, optional): Time of closest approach.
            y0 (float, optional): Minimum impact parameter.
            theta (float, optional): Inclination of source trajectory.
            centre_m2 (bool, optional): If True, t0/y0 refer to closest approach to primary lens (m2).
        """
        self.z1 = complex(d / 2.0, 0.0)  ## this is how we define a complex number in python!
        self.q = q
        self.dl = dl
        self.ds = ds
        m2 = m1 / q
        self.mtot = m1 + m2
        self.m1 = m1 / self.mtot
        self.m2 = m2 / self.mtot
        self.theta = theta

        """
        we build a point_lens instance to compute the einstein radius
        and the Einstein crossing time. This requires to define a
        point source instance too.
        """
        ps = point_source(ds=ds)
        pl = point_lens(ps=ps, mass=m1 + m2, dl=dl)
        self.thetaE = pl.EinsteinRadius()
        self.tE = pl.EinsteinCrossTime()

        if centre_m2:
            self.t02 = t0
            self.t0 = self.t0_from_t02()
            self.y02 = y0
            self.y0 = self.y0_from_y02()
        else:
            self.t0 = t0
            self.y0 = y0
            self.t02 = self.tM()
            self.y02 = self.yM()

    """
    This function finds the lens critical lines and caustics
    """

    def CritCau(self, ncpt=10000):
        """
        Compute the critical lines and caustics for the binary lens.

        Args:
            ncpt (int, optional): Number of phase points to sample. Default is 10000.

        Returns:
            tuple: (x, y, xs, ys) arrays for critical and caustic points.
        """
        # set the phase vector
        phi_ = np.linspace(0, 2. * np.pi, ncpt)

        x = []
        y = []
        xs = []
        ys = []

        # we need to find the roots of our fourth order polynomial for each value of phi
        for i in range(phi_.size):
            phi = phi_[i]
            # the coefficients of the complex polynomial
            coefficients = [1.0, 0.0, -2 * np.conj(self.z1) ** 2 - np.exp(1j * phi),
                            -np.conj(self.z1) * 2 * (self.m1 - self.m2) * np.exp(1j * phi),
                            np.conj(self.z1) ** 2 * (np.conj(self.z1) ** 2 - np.exp(1j * phi))]
            # use the numpy function roots to find the roots of the polynomial
            z = np.roots(coefficients)  # these are the critical points!

            # use the lens equation (complex form) to map the critical points
            # on the source plane
            zs = z - self.m1 / (np.conj(z) - np.conj(self.z1)) \
                 - self.m2 / ((np.conj(z) - np.conj(-self.z1)))  # these are the caustics!

            # append critical and caustic points
            x.append(z.real)
            y.append(z.imag)
            xs.append(zs.real)
            ys.append(zs.imag)

        return (np.array(x), np.array(y), np.array(xs), np.array(ys))

    """
    This function finds the images of a source at a given position with respect to the lens
    """

    def Images(self, ys1, ys2):
        """
        Find the images of a source at a given position with respect to the lens.

        Args:
            ys1 (float): Source x-coordinate.
            ys2 (float): Source y-coordinate.

        Returns:
            tuple: Arrays of image x and y coordinates.
        """
        zs = complex(ys1, ys2)
        m = 0.5 * (self.m1 + self.m2)
        Dm = (self.m2 - self.m1) / 2.0

        c5 = self.z1 ** 2 - np.conj(zs) ** 2
        c4 = -2 * m * np.conj(zs) + zs * np.conj(zs) ** 2 \
             - 2 * Dm * self.z1 - zs * self.z1 ** 2
        c3 = 4.0 * m * zs * np.conj(zs) + 4.0 * Dm * np.conj(zs) * self.z1 \
             + 2.0 * np.conj(zs) ** 2 * self.z1 ** 2 - 2.0 * self.z1 ** 4
        c2 = 4.0 * m ** 2 * zs + 4.0 * m * Dm * self.z1 \
             - 4.0 * Dm * zs * np.conj(zs) * self.z1 \
             - 2.0 * zs * np.conj(zs) ** 2 \
             * self.z1 ** 2 + 4.0 * Dm * self.z1 ** 3 \
             + 2.0 * zs * self.z1 ** 4
        c1 = -8.0 * m * Dm * zs * self.z1 \
             - 4.0 * Dm ** 2 * self.z1 ** 2 \
             - 4.0 * m ** 2 * self.z1 ** 2 \
             - 4.0 * m * zs * np.conj(zs) * self.z1 ** 2 \
             - 4.0 * Dm * np.conj(zs) * self.z1 ** 3 \
             - np.conj(zs) ** 2 * self.z1 ** 4 \
             + self.z1 ** 6
        c0 = self.z1 ** 2 * (4.0 * Dm ** 2 * zs \
                             + 4.0 * m * Dm * self.z1 \
                             + 4.0 * Dm * zs * np.conj(zs) * self.z1 + \
                             2.0 * m * np.conj(zs) * self.z1 ** 2 \
                             + zs * np.conj(zs) ** 2 * self.z1 ** 2 \
                             - 2 * Dm * self.z1 ** 3 - zs * self.z1 ** 4)

        coefficients = [c5, c4, c3, c2, c1, c0]

        images = np.roots(coefficients)
        """
        now, we need to drop the spurious solutions. This can be done 
        by checking which solutions satisfy the lens equation. 
        We compute the lens equation at the found solutions and check if 
        we recover the input source position.
        """
        z2 = -self.z1
        deltazs = zs - (images - self.m1 / (np.conj(images) - np.conj(self.z1))
                        - self.m2 / (np.conj(images) - np.conj(z2)))
        return (np.array([images.real[np.abs(deltazs) < 1e-3]]),
                np.array([images.imag[np.abs(deltazs) < 1e-3]]))

    def SourcePos(self, t):
        """
        Compute the source position at time t.

        Args:
            t (float): Time.

        Returns:
            tuple: (x, y) source coordinates.
        """
        p = (t - self.t0) / self.tE.value
        zreal = (np.cos(self.theta) * p - np.sin(self.theta) * self.y0)
        zimag = (np.sin(self.theta) * p + np.cos(self.theta) * self.y0)
        # z=complex(zreal,zimag)
        return (zreal, zimag)

    def detA(self, z):
        """
        Compute the determinant of the Jacobian matrix at position z.

        Args:
            z (complex or np.ndarray): Image position(s).

        Returns:
            float or np.ndarray: Determinant value(s).
        """
        z2 = -self.z1
        deta = 1 - np.abs(self.m1 / (np.conj(z) - np.conj(self.z1)) ** 2 + self.m2 / (np.conj(z) - np.conj(z2)) ** 2)
        return (deta)

    def Magnification(self, t):
        """
        Compute the total magnification at time t.

        Args:
            t (float): Time.

        Returns:
            float: Total magnification.
        """
        ys1, ys2 = self.SourcePos(t)
        xi1, xi2 = self.Images(ys1, ys2)
        images = xi1 + 1j * xi2
        mu = 1.0 / self.detA(images)
        return (np.abs(mu).sum())

    def LightCurve(self, times):
        """
        Compute the light curve for a sequence of times.

        Args:
            times (array-like): Sequence of times.

        Returns:
            tuple: (p, mu) arrays for normalized time and magnification.
        """
        p = (times - self.t0) / self.tE.value
        mu = []
        for t in times:
            mu.append(self.Magnification(t))
        return (p, mu)

    def MagnificationMap(self, xmin, xmax, ymin, ymax, npix=200):
        """
        Compute the magnification map over a grid in the source plane.

        Args:
            xmin (float): Minimum x value.
            xmax (float): Maximum x value.
            ymin (float): Minimum y value.
            ymax (float): Maximum y value.
            npix (int, optional): Number of pixels per axis (default 200).

        Returns:
            np.ndarray: Magnification map array.
        """
        xx = np.linspace(xmin, xmax, 200)
        yy = np.linspace(ymin, ymax, 200)
        X, Y = np.meshgrid(xx, yy)
        mu_arr = np.zeros_like(X)

        # z=X+1j*Y
        # deta=1.0/np.abs(self.detA(z))
        for j in range(len(xx)):
            for k in range(len(xx)):
                xi1, xi2 = self.Images(X[j, k], Y[j, k])
                images = xi1 + 1j * xi2
                mu = 1.0 / self.detA(images)
                mu_arr[j, k] = np.abs(mu).sum()
        return mu_arr

    def tM(self):
        """
        Compute the time of closest approach to lens m2.

        Returns:
            float: Time of closest approach to m2.
        """
        """
        Computes the time of closest approach to m2
        """
        A = np.cos(self.theta)
        tM = self.t0 - self.tE.value * (A * self.z1.real)
        return tM

    def yM(self):
        """
        Compute the minimum impact parameter to lens m2.

        Returns:
            float: Minimum impact parameter to m2.
        """
        xm, ym = self.SourcePos(self.t02)
        return np.sqrt((xm + self.z1.real) ** 2 + ym ** 2)

    def t0_from_t02(self):
        """
        Convert t02 (closest approach to m2) to t0 (midpoint).

        Returns:
            float: t0 value.
        """
        return (self.t02 + self.tE.value * np.cos(self.theta) * self.z1.real)

    def y0_from_y02(self):
        """
        Convert y02 (closest approach to m2) to y0 (midpoint).

        Returns:
            float: y0 value.
        """
        return self.y02 + self.z1.real * np.sin(self.theta)

    """
    Some utilities
    """

    def getPos(self):
        """
        Get the position of lens 1 (z1).

        Returns:
            complex: Position of lens 1.
        """
        return (self.z1)

    def gettE(self):
        """
        Get the Einstein crossing time.

        Returns:
            float: Einstein crossing time.
        """
        return (self.tE)

    def getThetaE(self):
        """
        Get the Einstein radius.

        Returns:
            float: Einstein radius.
        """
        return (self.thetaE)

    def WideIntTrans(self):
        """
        Compute the wide-intermediate transition separation.

        Returns:
            float: Wide-intermediate transition value.
        """
        dwi = ((self.m1) ** (1. / 3.) + (self.m2) ** (1. / 3.)) ** (3. / 2.)
        return (dwi)

    def IntCloseTrans(self):
        """
        Compute the intermediate-close transition separation.

        Returns:
            float: Intermediate-close transition value.
        """
        dic = ((self.m1) ** (1. / 3.) + (self.m2) ** (1. / 3.)) ** (-3. / 4.)
        return (dic)