from skimage import measure
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon
from shapely.ops import polygonize, unary_union
import numpy as np
from numpy.linalg import eig, inv


class map_obj(object):

    def __init__(self, img):
        """
        Initialize a map object using a numpy array.

        Args:
            img (np.ndarray): Input map (2D numpy array).
        """
        self.img = img

    def radial_profile(self, xcen=None, ycen=None, nbins=10, auto_center=True, rmin=None, rmax=None, logspace=False):
        """
        Compute the radial profile of the map.

        Args:
            xcen (float, optional): X-coordinate of center (pixels).
            ycen (float, optional): Y-coordinate of center (pixels).
            nbins (int, optional): Number of bins for the profile.
            auto_center (bool, optional): If True, center on image midpoint.
            rmin (float, optional): Minimum radius (pixels).
            rmax (float, optional): Maximum radius (pixels).
            logspace (bool, optional): If True, use logarithmic bins.

        Returns:
            tuple: (rbin, rb, radial_mean, radial_std)
        """
        rbin, rb, radial_mean, radial_std = None, None, None, None

        if auto_center:
            if rmin==None or rmax==None:
                sx, sy = self.img.shape
                X, Y = np.ogrid[0:sx, 0:sy]

                if logspace == False:
                    r = np.hypot(X - sx / 2, Y - sy / 2)
                    rbin = (nbins * r / r.max()).astype(int)
                    radial_mean = ndimage.mean(self.img, labels=rbin, index=np.arange(0, rbin.max()))
                    rb = ndimage.mean(r, labels=rbin, index=np.arange(0, rbin.max()))
                else:
                    r = np.sqrt((X - sx / 2)**2 + (Y - sy / 2)**2)
                    rmax=r.max()
                    rad_vec = np.logspace(0, np.log10(rmax), nbins)
                    rbin=np.zeros(self.img.shape,dtype=int)
                    for i in range(nbins-1):
                        rbin[(r >= rad_vec[i]) & (r < rad_vec[i+1])]=i+1
                        if (np.array([(r >= rad_vec[i]) & (r < rad_vec[i+1])]).sum()) == 0:
                            raise Exception('Bin size is too small! No elements in bin')
                    radial_mean = ndimage.mean(self.img, labels=rbin, index=np.arange(1, rbin.max()))
                    rb = ndimage.mean(r, labels=rbin, index=np.arange(1, rbin.max()))
            else:
                sx, sy = self.img.shape
                if (xcen == None or ycen == None):
                    xcen = sx/2.0
                    ycen = sy/2.0
                a = int(round(sx/2.0))
                b = int(round(sy/2.0))
                X, Y = np.ogrid[-a:sx-a, -b:sy-b]
                pix_x0 = round(xcen - sx / 2.0)
                pix_y0 = round(ycen - sy / 2.0)

                r = np.sqrt((X - pix_x0) ** 2 + (Y - pix_y0) ** 2)
                if logspace:
                    rad_vec = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
                else:
                    rad_vec = np.linspace(rmin, rmax, nbins)
                rbin = np.zeros(self.img.shape, dtype=int)
                for i in range(nbins - 1):
                    rbin[(r >= rad_vec[i]) & (r < rad_vec[i + 1])] = i+1
                    if (np.array([(r >= rad_vec[i]) & (r < rad_vec[i + 1])]).sum()) == 0:
                        raise Exception('Bin size is too small! No elements in bin')
                radial_mean = ndimage.mean(self.img, labels=rbin, index=np.arange(1, rbin.max()))
                radial_std = ndimage.standard_deviation(self.img, labels=rbin, index=np.arange(1, rbin.max()))
                rb = ndimage.mean(r, labels=rbin, index=np.arange(1, rbin.max()))

        return rbin, rb, radial_mean, radial_std

    def get_contours(self, lev, fully_connected='low'):
        """
        Extract contours from the map as CriticalLine objects.

        Args:
            lev (float): Contour level.
            fully_connected (str, optional): Connectivity for contour extraction (default 'low').

        Returns:
            list: List of CriticalLine objects, ordered by area size.
        """
        from pyLensLib.critcau import CriticalLine

        contours = measure.find_contours(self.img, lev,fully_connected=fully_connected)
        lc_all = []
        j = 0
        for contour in contours:
            contour[:, 0], contour[:, 1] = contour[:, 1].copy(), contour[:, 0].copy()
            ls = LineString(contour)
            lr = LineString(ls.coords[:] + ls.coords[0:1])
            mls = unary_union(lr)
            mp = MultiPolygon(list(polygonize(mls)))
            lc = CriticalLine(j, mp)
            lc.setPoints(contour)
            A = 0.0
            for g in range(len(mp.geoms)):
                A += mp.geoms[g].area
            lc.setArea(A)
            lc_all.append(lc)
            j += 1
        cl = sorted(lc_all, key=lambda x: x.getArea(), reverse=True)
        return (cl)

class contour_fit(object):
    """
    Fits ellipse to contour object
    """
    def __init__(self, cl):
        """
        Initialize contour fit object for ellipse fitting.

        Args:
            cl: Contour object, as coordinates [x, y].
        """

        self.x_list, self.y_list = zip(*cl.points)
        self.x_list = np.array(self.x_list)
        self.y_list = np.array(self.y_list)

    def fitEllipse(self):
        """
        Fit an ellipse to the contour using the direct least squares method.

        Returns:
            np.ndarray: Ellipse coefficient vector (axx, axy, ayy, ax, ay, a1).
        """

        x = self.x_list[:, np.newaxis]
        y = self.y_list[:, np.newaxis]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        E, V = eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        self.a = V[:, n]


    def ellipse_center(self):
        """
        Find the center of the best-fit ellipse.

        Returns:
            np.ndarray: Coordinates of the ellipse center [x0, y0].
        """
        b, c, d, f, g, a = self.a[1] / 2, self.a[2], self.a[3] / 2, self.a[4] / 2, self.a[5], self.a[0]
        num = b * b - a * c
        x0 = (c * d - b * f) / num
        y0 = (a * f - b * d) / num
        return np.array([x0, y0])

    def ellipse_angle_of_rotation(self):
        """
        Find the orientation (position angle) of the best-fit ellipse.

        Returns:
            float: Ellipse position angle (radians).
        """
        b, c, d, f, g, a = self.a[1] / 2, self.a[2], self.a[3] / 2, self.a[4] / 2, self.a[5], self.a[0]
        return 0.5 * np.arctan(2 * b / (a - c))

    def ellipse_axis_length(self):
        """
        Find the semi-major and minor axes of the best-fit ellipse.

        Returns:
            np.ndarray: Semi-major and minor axes of the ellipse.
        """
        b, c, d, f, g, a = self.a[1] / 2, self.a[2], self.a[3] / 2, self.a[4] / 2, self.a[5], self.a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        res1 = np.sqrt(abs(up / down1))
        res2 = np.sqrt(abs(up / down2))
        return np.array([res1, res2])

class image_fit(object):
    """
    Analyze image properties using intensity moments.

    This class computes the centroid, quadrupole moments, ellipticity, and axis lengths
    of a 2D image above a given intensity threshold. It is useful for characterizing
    the shape and orientation of objects in astronomical or other scientific images.

    Attributes:
        img (numpy.ndarray): Input 2D image array.
        ith (float): Intensity threshold for pixel selection.
        img_sel (numpy.ndarray): Flattened array of selected pixel intensities.
        r (numpy.ndarray): Coordinates of selected pixels.
        flux (float): Total flux (sum of selected intensities).
        c (list): Centroid coordinates [x, y].
        Q (numpy.ndarray): 2x2 quadrupole moment matrix.
    """

    def __init__(self,img,ith=0):
        """
        Initialize the image_fit object and compute key image properties above a threshold.

        Args:
            img (numpy.ndarray): Input 2D image array.
            ith (float, optional): Intensity threshold for pixel selection. Defaults to 0.

        Attributes:
            img (numpy.ndarray): The input image.
            ith (float): The intensity threshold.
            img_sel (numpy.ndarray): Flattened array of selected pixel intensities above threshold.
            r (numpy.ndarray): Coordinates of selected pixels.
            flux (float): Total flux (sum of selected intensities).
            c (list): Centroid coordinates [x, y].
            Q (numpy.ndarray): 2x2 quadrupole moment matrix.
        """
        self.img=img
        self.ith=ith
        isel= img > ith
        self.img_sel=self.img[isel].flatten()
        theta1 = np.linspace(0, img.shape[0] - 1 ,img.shape[0])
        theta2 = np.linspace(0, img.shape[0] - 1, img.shape[0])
        x_,y_=np.meshgrid(theta1,theta2)
        self.r = np.stack((x_[isel].flatten(), y_[isel].flatten()), axis=1)
        self.flux=self.img_sel.sum()
        self.c=self.centroid()
        self.Q=self.quadrupoles()

    def centroid(self):
        """
        Compute the centroid (center of mass) of the selected image region.

        The centroid is calculated as the intensity-weighted mean position of the selected pixels.

        Returns:
            list: Coordinates [x, y] of the centroid.
        """
        c=[np.sum(self.img_sel*self.r[:,i])/np.sum(self.img_sel) for i in range(2)]
        return c

    def quadrupoles(self):
        """
        Compute the normalized quadrupole moment matrix of the image.

        The quadrupole moments characterize the second-order spatial distribution of the image intensity,
        centered at the centroid. The result is a 2x2 matrix normalized by the total flux.

        Returns:
            numpy.ndarray: 2x2 quadrupole moment matrix.
        """
        Q=np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                Q[i,j]=np.sum(self.img_sel*(self.r[:,i]-self.c[i])*(self.r[:,j]-self.c[j]))
        Q/=self.flux
        return Q

    def ellipticity(self):
        """
        Compute the image ellipticity from the quadrupole moments.

        The ellipticity components (e1, e2) are calculated using the normalized quadrupole matrix.
        The sign of the ellipticity is determined by the ordering of the axes (major vs. minor).

        Returns:
            tuple: (e1, e2) ellipticity components. Returns (0.0, 0.0) if the denominator is zero.
        """
        a, b = self.axes()
        Q = self.Q
        denom = Q[0, 0] + Q[1, 1] + 2 * np.sqrt(Q[0, 0] * Q[1, 1] - Q[0, 1] ** 2)
        # Avoid division by zero
        if denom == 0:
            return 0.0, 0.0
        e1 = (Q[0, 0] - Q[1, 1]) / denom
        e2 = 2 * Q[0, 1] / denom
        if a > b:
            return e1, e2
        else:
            return -e1, -e2

    def axes(self):
        """
        Compute the eigenvalues of the quadrupole moment matrix.

        These eigenvalues correspond to the squared lengths of the semi-major and semi-minor axes
        of the image's light distribution.

        Returns:
            tuple: (l1, l2) where l1 and l2 are the eigenvalues (axis lengths squared).
        """
        l1=0.5*(self.Q[0,0]+self.Q[1,1]+np.sqrt((self.Q[0,0]-self.Q[1,1])**2+4.0*self.Q[0,1]**2))
        l2=0.5*(self.Q[0,0]+self.Q[1,1]-np.sqrt((self.Q[0,0]-self.Q[1,1])**2+4.0*self.Q[0,1]**2))
        return l1, l2 #1.0/np.sqrt(l1),1.0/np.sqrt(l2)






