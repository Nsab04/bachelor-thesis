import numpy as np
import matplotlib.pyplot as plt
try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - optional dependency
    gpd = None
from shapely.geometry import Point, Polygon
import pandas as pd

def Random_Points_in_Bounds(polygon, number):
    """
    Generate random points within the bounds of a given polygon.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to sample points in.
        number (int): Number of random points to generate.

    Returns:
        tuple: Arrays of x and y coordinates of random points.
    """
    minx, miny, maxx, maxy = polygon.bounds
    x = np.random.uniform(minx, maxx, number)
    y = np.random.uniform(miny, maxy, number)
    return x, y

class CriticalLine(object):
    """
    Represents a critical line in gravitational lensing analysis.

    Attributes:
        ID (int): Identifier for the critical line.
        geometria (shapely.geometry.Polygon): Geometry of the critical line.
        principale (bool): Flag for main critical line.
        passsize (bool): Flag for size pass.
        sigmaloc (float): Local sigma value.
        area (float): Area enclosed by the critical line.
        points (list): List of (x, y) points on the critical line.
        px, py (float): Mean x and y positions of the critical line.
    """
    def __init__(self, ID, geometria):
        """
        Initialize a CriticalLine instance.

        Args:
            ID (int): Identifier for the critical line.
            geometria (Polygon): Geometry of the critical line.
        """
        self.ID = int(ID)
        self.principale = False
        self.passsize = False
        self.geometria = geometria
        self.sigmaloc = 10000.0

    def setArea(self, area):
        """
        Set the area enclosed by the critical line.

        Args:
            area (float): Area value.
        """
        self.area = area

    def setPoints(self, points):
        """
        Set the points defining the critical line and compute mean position.

        Args:
            points (list): List of (x, y) tuples.
        """
        self.points = points
        c1, c2 = zip(*points)
        c1 = np.array(c1)
        c2 = np.array(c2)
        self.px = c1.mean()
        self.py = c2.mean()

    def showCC(self):
        """
        Plot the critical line and its mean position if enough points are present.
        """
        c1, c2 = zip(*self.points)
        c1 = np.array(c1)
        c2 = np.array(c2)
        if (c1.size > 100):
            ftest, axtest = plt.subplots(1, 1, figsize=(10, 10))
            axtest.plot(c1, c2, '-')
            axtest.plot([self.px], [self.py], 'o')
            axtest.set_aspect('equal')
            plt.show()
            plt.close(ftest)

    def getArea(self):
        """
        Get the area enclosed by the critical line.

        Returns:
            float: Area value.
        """
        return self.area

    def getThetaE(self):
        """
        Compute the Einstein radius from the area.

        Returns:
            float: Einstein radius.
        """
        return np.sqrt(self.area/np.pi)

    def toArcsec(self, npix, pixel_scale):
        """
        Convert the critical line points to arcseconds.

        Args:
            npix (int): Number of pixels.
            pixel_scale (float): Pixel scale in arcseconds.
        """
        vs = self.points
        x1, x2 = zip(*vs)
        x1 = (np.array(x1) - npix/2.0) * pixel_scale
        x2 = (np.array(x2) - npix/2.0) * pixel_scale
        self.points = zip(x1, x2)

class Caustic(object):
    """
    Represents a caustic curve in gravitational lensing analysis.

    Attributes:
        ID (int): Identifier for the caustic.
        geometria (shapely.geometry.Polygon): Geometry of the caustic.
        principale (bool): Flag for main caustic.
        passsize (bool): Flag for size pass.
        sigmaloc (float): Local sigma value.
        with_buffer (bool): Flag for buffer presence.
        area (float): Area enclosed by the caustic.
        points (list): List of (x, y) points on the caustic.
        buffer (Polygon): Buffer geometry.
        buffer_area (float): Area of the buffer.
    """
    def __init__(self, ID, geometria):
        """
        Initialize a Caustic instance.

        Args:
            ID (int): Identifier for the caustic.
            geometria (Polygon): Geometry of the caustic.
        """
        self.ID = ID
        self.geometria = geometria
        self.principale = False
        self.passsize = False
        self.sigmaloc = 10000.0
        self.with_buffer = False

    def setArea(self, area):
        """
        Set the area enclosed by the caustic.

        Args:
            area (float): Area value.
        """
        self.area = area

    def setPoints(self, points):
        """
        Set the points defining the caustic.

        Args:
            points (list): List of (x, y) tuples.
        """
        self.points = points

    def addBuffer(self, buffer_size):
        """
        Add a buffer around the caustic.

        Args:
            buffer_size (float): Buffer size in pixels.
        """
        self.with_buffer = True
        if buffer_size > 0.0:
            self.buffer = Polygon(self.geometria.buffer(buffer_size))
            self.buffer_area = self.buffer.area
        else:
            self.buffer = Polygon(self.geometria.buffer(1e-4))
            self.buffer_area = self.buffer.area

    def getArea(self):
        """
        Get the area enclosed by the caustic.

        Returns:
            float: Area value.
        """
        return self.area

    def getBufferArea(self):
        """
        Get the area of the buffer around the caustic.

        Returns:
            float: Buffer area if buffer exists, else None.
        """
        if self.with_buffer:
            return self.buffer_area

    def toArcsec(self, npix, pixel_scale):
        """
        Convert the caustic points to arcseconds.

        Args:
            npix (int): Number of pixels.
            pixel_scale (float): Pixel scale in arcseconds.
        """
        self.points = (self.points - npix/2.0 + 0.5) * pixel_scale

    def random_points_in_caustic(self, buffer_size=0.0, number=10000):
        """
        Randomly place points inside a caustic (or a caustic with outer buffer).

        Args:
            buffer_size (float): Buffer size around the caustic.
            number (int): Number of random points to generate.

        Returns:
            GeoDataFrame: Points inside the caustic region.
        """
        if gpd is None:
            raise ImportError("geopandas is required for random_points_in_caustic().")
        self.addBuffer(buffer_size=buffer_size)
        x, y = Random_Points_in_Bounds(self.buffer, number)
        gdf_poly = gpd.GeoDataFrame(index=["myPoly"], geometry=[self.buffer])
        df = pd.DataFrame()
        df['points'] = list(zip(x, y))
        df['points'] = df['points'].apply(Point)
        gdf_points = gpd.GeoDataFrame(df, geometry='points')
        Sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate="within", how='left')
        pnts_in_caustic = gdf_points[Sjoin.index_right == 'myPoly']
        return pnts_in_caustic

    def random_points_along_caustic(self, buffer_size=0.0, number=10000):
        """
        Randomly place points along a buffer region around the caustic.

        Args:
            buffer_size (float): Buffer size around the caustic.
            number (int): Number of random points to generate.

        Returns:
            GeoDataFrame: Points randomly drawn in a buffer along the caustic.
        """
        if gpd is None:
            raise ImportError("geopandas is required for random_points_along_caustic().")
        cau_geom = self.geometria
        c1 = cau_geom.buffer(buffer_size)
        c2 = cau_geom.buffer(-buffer_size)
        cc = c1.difference(c2)

        x, y = Random_Points_in_Bounds(self.buffer, number)
        gdf_poly = gpd.GeoDataFrame(index=["myPoly"], geometry=[cc])
        df = pd.DataFrame()
        df['points'] = list(zip(x, y))
        df['points'] = df['points'].apply(Point)
        gdf_points = gpd.GeoDataFrame(df, geometry='points')
        Sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate="within", how='left')
        pnts_along_caustic = gdf_points[Sjoin.index_right == 'myPoly']
        return pnts_along_caustic
