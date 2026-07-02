from scipy.ndimage import map_coordinates
import shapely.geometry
from shapely.ops import polygonize, unary_union
from skimage import measure
from pyLensLib.critcau import CriticalLine
import numpy as np

def sort_contours(contours):
    """
    Sort and process a list of contour arrays into CriticalLine objects, ordered by area.

    Args:
        contours (list): List of contour arrays from skimage.measure.find_contours.

    Returns:
        list: Sorted list of CriticalLine objects (largest area first).
    """
    c_all = []
    j = 0
    for contour in contours:
        contour[:, 0], contour[:, 1] = contour[:, 1].copy(), contour[:, 0].copy()
        ls = shapely.geometry.LineString(contour)
        lr = shapely.geometry.LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(lr)
        mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
        c = CriticalLine(j, mp)
        c.setPoints(contour)
        A = 0.0
        for g in range(len(mp.geoms)):
            A += mp.geoms[g].area
        c.setArea(A)
        c_all.append(c)
        j += 1
    c_sorted = sorted(c_all, key=lambda x: x.getArea(), reverse=True)
    return (c_sorted)

class gensrc(object):
    def __init__(self) -> object:
        """
        Initialize a gensrc object for source modeling and ray tracing.
        """
        self.initialized=True

    def ray_trace(self):
        """
        Perform ray tracing from the image plane to the source plane using the deflection field.

        Returns:
            tuple: (y1, y2) coordinates on the source plane.
        """
        # px = self.df.grid_pixel  # size/(self.df.npix-1)
        px = self.df.pixel_scale

        # x1pix = (self.x1 + (self.df.size1) / 2.0) / px
        # x2pix = (self.x2 + (self.df.size2) / 2.0) / px

        x1pix = (self.x1 - self.df.thetax[0]) / px
        x2pix = (self.x2 - self.df.thetay[0]) / px

        if len(x1pix.shape) > 1:
            x1pix[:, -1] = np.round(x1pix[:, -1], 0)
            x2pix[-1, :] = np.round(x2pix[-1, :], 0)
            x1pix[:, 0] = np.round(x1pix[:, 0], 0)
            x2pix[0, :] = np.round(x2pix[0, :], 0)
        else:
            x1pix[-1] = np.round(x1pix[-1], 0)
            x2pix[-1] = np.round(x2pix[-1], 0)
            x1pix[0] = np.round(x1pix[0], 0)
            x2pix[0] = np.round(x2pix[0], 0)

        #if np.round(x1pix[0].take(-1), 6) > self.df.a1.shape[1] - 1 \
        #        or np.round(x2pix[-1].take(0), 6) > self.df.a1.shape[0] - 1 \
        #        or np.round(x1pix[0].take(0), 6) < 0 \
        #        or np.round(x2pix[0].take(0), 6) < 0:
        #    raise Exception('Tracing rays outside of deflection field')

        # if (self.df.size < self.size):
        #     raise Exception('The size of the deflector is too small compared '
        #                     'to the size of the postage stamp')

        a1 = map_coordinates(self.df.a1,
                             [x2pix, x1pix], order=1, prefilter=True)
        a2 = map_coordinates(self.df.a2,
                             [x2pix, x1pix], order=1, prefilter=True)


        if self.df.perturbed:
            # compute elements of the Jacobian matrix
            a11 = 1.0 - self.df.pb.pkappa - self.df.pb.pgamma1
            a22 = 1.0 - self.df.pb.pkappa + self.df.pb.pgamma1
            a12 = -self.df.pb.pgamma2
            # compute elements of tensor D
            a111 = -0.5 * (self.df.pb.pg1 + 3.0 * self.df.pb.pf1)
            a222 = -0.5 * (3.0 * self.df.pb.pf2 - self.df.pb.pg2)
            a112 = -0.5 * (self.df.pb.pf2 + self.df.pb.pg2)
            a221 = -0.5 * (self.df.pb.pf1 - self.df.pb.pg1)

            z1 = self.x1 - self.ys1 - a1 * self.rescf - self.df.pb.pa1 * self.rescf
            z2 = self.x2 - self.ys2 - a2 * self.rescf - self.df.pb.pa2 * self.rescf

            k1 = a11 * (z1) + a12 * (z2) + \
                 0.5 * a111 * (z1) ** 2 + a112 * (z1) * (z2) + \
                 0.5 * a221 * (z2) ** 2
            k2 = a22 * (z2) + a12 * (z1) + \
                 0.5 * a222 * (z2) ** 2 + a221 * (z1) * (z2) + \
                 0.5 * a112 * (z1) ** 2

            y1 = k1 + self.ys1  # y1 coordinates on the source plane
            y2 = k2 + self.ys2  # y2 coordinates on the source plane

        else:
            y1 = (self.x1 - a1 * self.rescf)  # y1 coordinates on the source plane
            y2 = (self.x2 - a2 * self.rescf)  # y2 coordinates on the source plane

        return (y1, y2)

    def image_contours(self, level=0.0):
        """
        Find and sort contours in the lensed image at a given level.

        Args:
            level (float, optional): Contour level (default is 0.0).

        Returns:
            list: Sorted list of CriticalLine objects for the image contours.
        """

        contours = measure.find_contours(self.image, level)
        c_sorted = sort_contours(contours)
        return(c_sorted)

    def source_contours(self, level=0.0):
        """
        Find and sort contours in the unlensed source image at a given level.

        Args:
            level (float, optional): Contour level (default is 0.0).

        Returns:
            list: Sorted list of CriticalLine objects for the source contours.
        """

        assert self.save_unlensed, "The unlensed image was not saved! Define source with save_unlensed=True"
        contours = measure.find_contours(self.image_unlensed, level)
        c_sorted = sort_contours(contours)
        return (c_sorted)

    def getContourPoints(self, lc, px=1.0, pixel_units=False):
        """
        Given a CriticalLine object, return lists of critical point coordinates.

        Args:
            lc (CriticalLine): Critical line object.
            px (float, optional): Pixel scale (default is 1.0).
            pixel_units (bool, optional): If True, return pixel units; else, convert to physical units.

        Returns:
            tuple: (x1, x2) lists of critical point coordinates.
        """

        vs=lc.points
        x1,x2=zip(*vs)
        if (pixel_units):
            return(x1,x2)
        else:
            x1=(np.array(x1)-self.image.shape[1]/2.0)*px
            x2=(np.array(x2)-self.image.shape[0]/2.0)*px
            return(x1,x2)





