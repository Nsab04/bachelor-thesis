from pyLensLib.gensrc import gensrc
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import root


def same_sign3(a, b, c, eps=1e-12):
    """
    Check if three values have the same sign, considering a tolerance.

    Args:
        a, b, c (float or array): Values to compare.
        eps (float): Tolerance for zero.

    Returns:
        bool or array: True if all have the same sign or are close to zero.
    """
    return ((a * b > 0) & (b * c > 0)) | ((abs(a) < eps) & (abs(b) < eps) & (abs(c) < eps))

class pointsrc(gensrc):

    """
    Class representing a point source for lensing simulations.

    Attributes:
        ys1, ys2 (float): Source position coordinates.
        flux (float): Source flux.
        zs (float): Source redshift.
        refine (bool): Whether to refine image positions.
        refine_to_td (bool): Refine image positions to TD stationary points.
        rescf (float): Rescaling factor for lensing geometry.
        N (int): Number of pixels.
        df: Deflector object.
        size (float): Image size.
        center_frame (list): Center coordinates of the frame.
        x1, x2 (ndarray): Meshgrid coordinates.
        pixel (float): Pixel scale.
        image (ndarray): Lensed image.
        image_unlensed (ndarray): Unlensed image.
        xi1_raw, xi2_raw (ndarray): Raw image positions before TD refinement.
        td_refine_shift (ndarray): Per-image refinement shifts in arcsec.
        td_refine_success (ndarray): Boolean flags for TD refinement success.
    """
    def __init__(self, size=100.0, sizex=None, sizey=None, Npix=100, gl=None,
                 save_unlensed=False, use_lenstronomy=False, refine=False, refine_to_td=False,
                 td_surface=None, td_grad_x=None, td_grad_y=None, **kwargs):
        """
        Initialize a point source object for lensing simulation.

        Args:
            size (float): Image size.
            sizex, sizey (tuple, optional): x/y range for custom frame.
            Npix (int): Number of pixels.
            gl: Deflector object.
            save_unlensed (bool): Save unlensed image if True.
            use_lenstronomy (bool): Use lenstronomy for image finding if True.
            refine (bool): Refine image positions if True.
            refine_to_td (bool): Refine image positions to TD stationary points
                after the regular image finder has produced candidate positions.
            td_surface (ndarray, optional): Precomputed time-delay surface to use
                for TD refinement.
            td_grad_x, td_grad_y (ndarray, optional): Precomputed TD surface
                gradients along x/y in arcsec units.
            **kwargs: Source parameters (ys1, ys2, flux, zs).

        Returns:
            None
        """
        super().__init__()
        if ('ys1' in kwargs):
            self.ys1 = kwargs['ys1']
        else:
            self.ys1 = 0.0

        if ('ys2' in kwargs):
            self.ys2 = kwargs['ys2']
        else:
            self.ys2 = 0.0

        if ('flux' in kwargs):
            self.flux = kwargs['flux']
        else:
            self.flux = 100.0

        if ('zs' in kwargs):
            self.zs = kwargs['zs']
        else:
            self.zs = 1.0

        self.refine = refine
        self.refine_to_td = refine_to_td
        self.td_surface = td_surface
        self.td_grad_x = td_grad_x
        self.td_grad_y = td_grad_y

        self.rescf = 1.0
        if gl != None:
            if self.zs != gl.zs:
                if self.zs > gl.zl:
                    ds = gl.co.angular_diameter_distance(self.zs).value
                    dls = gl.co.angular_diameter_distance_z1z2(gl.zl,self.zs).value
                    self.rescf=dls/ds*gl.ds/gl.dls
                else:
                    self.rescf = 0.0

        self.N = Npix
        self.df = gl

        # define the pixel coordinates
        if sizex == None or sizey == None:
            self.size = float(size)
            pcx = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
            pcy = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
            self.center_frame = [0.,0.]
        else:
            pcx = np.linspace(sizex[0], sizex[1], self.N)
            pcy = np.linspace(sizey[0], sizey[1], self.N)
            self.size=sizex[1]-sizex[0]
            self.center_frame = [(sizex[0] + sizex[1]) / 2.0, (sizey[0] + sizey[1]) / 2.0]
        #pc = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)

        self.x1, self.x2 = np.meshgrid(pcx, pcy)

        self.pixel = self.size / (self.N - 1)

        if self.df == None: # NO LENS
            self.image = self.brightness(self.ys1,self.ys2)
            if(save_unlensed):
                self.image_unlensed = self.image
        else:               # LENS
            if(save_unlensed):
                self.image_unlensed = self.brightness(self.ys1,self.ys2)
            self.image = np.zeros((self.N, self.N))
            if use_lenstronomy:
                self.xi1, self.xi2, self.mui = self.find_images_lenstronomy()
            else:
                self.xi1, self.xi2, self.mui = self.find_images()
            self.xi1_raw = np.array(self.xi1, copy=True)
            self.xi2_raw = np.array(self.xi2, copy=True)
            if self.refine_to_td and self.df is not None and len(self.xi1) > 0:
                self.xi1, self.xi2, self.mui = self.refineImages_TD(self.xi1, self.xi2)
            #print (self.xi1)
            #print (self.xi2)
            #print (self.mui)
            for i in range(len(self.xi1)):
                image_tmp = self.brightness(self.xi1[i],self.xi2[i], self.mui[i])
                self.image=self.image+image_tmp.copy()


    def brightness(self,ys1,ys2,mu=1.0):
        """
        Generate a brightness map for a point source at a given position.

        Args:
            ys1, ys2 (float): Source position coordinates.
            mu (float): Magnification factor.

        Returns:
            ndarray: Brightness map.
        """
        # convert image positions with respect to the deflector center into 
        # image positions with respect to the image center
        #print (self.center_frame)
        ys1 = ys1 - self.center_frame[0]
        ys2 = ys2 - self.center_frame[1]

        pix = self.pixel

        px = int(ys1/pix + self.N/2.)
        py = int(ys2/pix + self.N/2.)
        brightness=np.zeros((self.N,self.N))
        if ((px >= 0) & (px<self.N) & (py >= 0) & (py < self.N)):
            brightness[py,px]=self.flux*mu
        return (brightness)

    """
    NEW IMPLEMENTATION OF FIND_IMAGES
    """

    def find_images(self):
        """
        Find image positions of the source using triangle mapping and refinement.

        Returns:
            tuple: Arrays of image x positions, y positions, and magnifications.
        """
        # --- Source position in pixel coordinates ---
        y1s = self.ys1 / self.df.pixel_scale + (len(self.df.thetax) - 1) / 2.0
        y2s = self.ys2 / self.df.pixel_scale + (len(self.df.thetay) - 1) / 2.0


        # --- Ray tracing: deflection ---
        y1 = self.df.theta1 - self.df.a1 * self.rescf
        y2 = self.df.theta2 - self.df.a2 * self.rescf
        # convert to pixel units
        xray = y1.copy() / self.df.pixel_scale + (len(self.df.thetax) - 1) / 2.0
        yray = y2.copy() / self.df.pixel_scale + (len(self.df.thetay) - 1) / 2.0

        # --- Create shifted maps for triangle corners ---
        xray1 = np.roll(xray, 1, axis=1)
        xray2 = np.roll(xray1, 1, axis=0)
        xray3 = np.roll(xray2, -1, axis=1)

        yray1 = np.roll(yray, 1, axis=1)
        yray2 = np.roll(yray1, 1, axis=0)
        yray3 = np.roll(yray2, -1, axis=1)

        # --- Compute offsets to source position for triangle vertices ---
        x1 = y1s - xray
        y1_ = y2s - yray
        x2 = y1s - xray1
        y2_ = y2s - yray1
        x3 = y1s - xray2
        y3 = y2s - yray2
        x4 = y1s - xray3
        y4 = y2s - yray3

        # --- Triangle inclusion tests using cross-products ---
        prod12 = x1 * y2_ - x2 * y1_
        prod23 = x2 * y3 - x3 * y2_
        prod31 = x3 * y1_ - x1 * y3
        prod13 = -prod31
        prod34 = x3 * y4 - x4 * y3
        prod41 = x4 * y1_ - x1 * y4

        mask1 = (np.sign(prod12) == np.sign(prod23)) & (np.sign(prod23) == np.sign(prod31))
        mask2 = (np.sign(prod13) == np.sign(prod34)) & (np.sign(prod34) == np.sign(prod41))

        image = np.zeros_like(xray, dtype=np.uint8)
        image[mask1] = 1
        image[mask2] = 2

        xi_all, yi_all = [], []

        for typ in [1, 2]:
            y_idx, x_idx = np.where(image == typ)
            # Filter out invalid or edge pixels
            valid = (x_idx > 0) & (y_idx > 0)
            x = x_idx[valid] - 1
            y = y_idx[valid] - 1

            if x.size == 0:
                continue

            if typ == 1:
                dx = np.array([x1[y, x], x2[y, x], x3[y, x]])
                dy = np.array([y1_[y, x], y2_[y, x], y3[y, x]])
            else:
                dx = np.array([x1[y, x], x3[y, x], x4[y, x]])
                dy = np.array([y1_[y, x], y3[y, x], y4[y, x]])

            # Compute weights based on distances
            w = 1. / np.sqrt(dx ** 2 + dy ** 2 + 1e-12)

            # Refine image positions using weighted triangle vertex average
            xi_ref, yi_ref = self.refineImagePositions(x, y, w, typ)
            #xi_all.append(xi_ref-1)
            #yi_all.append(yi_ref-1)
            xi_all.append(xi_ref)
            yi_all.append(yi_ref)

        if not xi_all:
            return np.array([]), np.array([]), np.array([])

        # Concatenate image positions
        xi = np.concatenate(xi_all)
        yi = np.concatenate(yi_all)

        # Newton-Raphson refinement (in pixel coordinates)
        #print("Before Newton:")
        #for i in range(len(xi)):
        #    print(f"({xi[i]:.5f}, {yi[i]:.5f})")


        # Refine one representative per cluster

        if self.refine:
            xi, yi = self.refineImages_Hybrid(xi, yi)


        #print("After Newton:")
        #for i in range(len(xi)):
        #    print(f"({xi[i]:.5f}, {yi[i]:.5f})")

        # Compute magnification at final positions
        mui = abs(self.mu_image(xi, yi))

        # Convert to physical units (arcsec)
        #xi = (xi - (len(self.df.thetax) - 1) / 2.0 +1) * self.df.pixel_scale
        #yi = (yi - (len(self.df.thetay) - 1) / 2.0 +1) * self.df.pixel_scale

        # adopting the convention that the triangle vertex coordinates are aligned with the lower-left corner of a pixel
        xi = (xi - (len(self.df.thetax) - 1) / 2.0) * self.df.pixel_scale
        yi = (yi - (len(self.df.thetay) - 1) / 2.0) * self.df.pixel_scale


        #xi = (xi - (len(self.df.thetax)) / 2.0) * self.df.pixel_scale
        #yi = (yi - (len(self.df.thetay)) / 2.0) * self.df.pixel_scale

        #xi_arcsec = (xi - (len(self.df.thetax) - 1) / 2.0) * self.df.pixel_scale
        #yi_arcsec = (yi - (len(self.df.thetay) - 1) / 2.0) * self.df.pixel_scale

        return xi, yi, mui

    def refineImages_TD(self, xi, yi):
        """
        Refine image positions by minimizing the gradient of the time-delay surface.

        Args:
            xi, yi (array): Image positions in arcseconds.

        Returns:
            tuple: Refined x and y positions in arcseconds and magnifications at the refined points.
        """
        xi = np.asarray(xi, dtype=float)
        yi = np.asarray(yi, dtype=float)
        if self.df is None or xi.size == 0:
            self.td_refine_shift = np.array([], dtype=float)
            self.td_refine_success = np.array([], dtype=bool)
            return xi, yi, np.asarray(self.mui, dtype=float)

        if self.td_surface is not None:
            td = np.asarray(self.td_surface, dtype=float)
        else:
            td = self.df.t_delay_surf(beta=(self.ys1, self.ys2))
            self.td_surface = td

        if self.td_grad_x is not None and self.td_grad_y is not None:
            grad_x = np.asarray(self.td_grad_x, dtype=float)
            grad_y = np.asarray(self.td_grad_y, dtype=float)
        else:
            grad_y, grad_x = np.gradient(td, self.df.pixel_scale, self.df.pixel_scale)
            self.td_grad_x = grad_x
            self.td_grad_y = grad_y

        refined_x = np.empty_like(xi)
        refined_y = np.empty_like(yi)
        shifts = np.empty_like(xi)
        success = np.zeros_like(xi, dtype=bool)

        def td_grad(v):
            xpix = float(v[0]) / self.df.pixel_scale + (self.df.nray1 - 1) / 2.0
            ypix = float(v[1]) / self.df.pixel_scale + (self.df.nray2 - 1) / 2.0
            gx = float(map_coordinates(grad_x, [[ypix], [xpix]], order=1, mode="nearest")[0])
            gy = float(map_coordinates(grad_y, [[ypix], [xpix]], order=1, mode="nearest")[0])
            return np.array([gx, gy], dtype=float)

        for i, (x0, y0) in enumerate(zip(xi, yi)):
            sol = root(td_grad, np.array([x0, y0], dtype=float), method="hybr")
            if sol.success and np.all(np.isfinite(sol.x)):
                refined_x[i] = float(sol.x[0])
                refined_y[i] = float(sol.x[1])
                success[i] = True
            else:
                refined_x[i] = x0
                refined_y[i] = y0
            shifts[i] = float(np.hypot(refined_x[i] - x0, refined_y[i] - y0))

        xpix = refined_x / self.df.pixel_scale + (len(self.df.thetax) - 1) / 2.0 + 1.0
        ypix = refined_y / self.df.pixel_scale + (len(self.df.thetay) - 1) / 2.0 + 1.0
        mui = np.abs(self.mu_image(xpix, ypix))

        self.td_refine_shift = shifts
        self.td_refine_success = success
        return refined_x, refined_y, mui

    def refineImagePositions(self, x, y, w, typ):
        """
        Refine image positions as weighted mean of triangle vertices.

        Args:
            x, y (array): Pixel coordinates.
            w (array): Weights for vertices.
            typ (int): Triangle type (1 or 2).

        Returns:
            tuple: Refined x and y positions.
        """
        """
        Refine image positions as weighted mean of triangle vertices.
        """
        if typ == 2:
            xp = np.array([x, x + 1, x + 1])
            yp = np.array([y, y, y + 1])
        else:
            xp = np.array([x, x + 1, x])
            yp = np.array([y, y + 1, y + 1])

        denom = np.sum(1. / w, axis=0)
        xi = np.sum(xp / w, axis=0) / denom
        yi = np.sum(yp / w, axis=0) / denom

        return xi, yi


    '''
    Find the images of a source at (ys1,ys2) by mapping triangles on the lens plane into
    triangles in the source plane. Then search for the triangles which contain the source. 
    The image position is then computed by weighing with the distance from the vertices of the 
    triangle on the lens plane
    '''

    def find_images_old(self):
        """
        Legacy method to find image positions by triangle mapping.

        Returns:
            tuple: Arrays of image x positions, y positions, and magnifications.
        """
        # map the source position in pixels onto the deflector grid
        #y1s = (self.ys1 + self.df.size / 2.0) / self.df.grid_pixel
        #y2s = (self.ys2 + self.df.size / 2.0) / self.df.grid_pixel
        y1s = self.ys1/self.df.pixel_scale + (len(self.df.thetax)-1) / 2.0
        y2s = self.ys2/self.df.pixel_scale + (len(self.df.thetay)-1) / 2.0


        # ray-trace the deflector grid onto the source plane
        y1 = self.df.theta1 - self.df.a1*self.rescf
        y2 = self.df.theta2 - self.df.a2*self.rescf

        # convert to pixel units
        xray = y1.copy() / self.df.pixel_scale + (len(self.df.thetax)-1) / 2.0
        yray = y2.copy() / self.df.pixel_scale + (len(self.df.thetay)-1) / 2.0

        # shift the maps by one pixel
        xray1 = np.roll(xray,   1, axis=1)
        xray2 = np.roll(xray1,  1, axis=0)
        xray3 = np.roll(xray2, -1, axis=1)
        yray1 = np.roll(yray,   1, axis=1)
        yray2 = np.roll(yray1,  1, axis=0)
        yray3 = np.roll(yray2, -1, axis=1)

        """
        For each pixel on the LENS plane, build two triangles. By means of 
        ray-tracing these are mapped onto the source plane into other two 
        triangles. Compute the distances of the vertices of the triangles on 
        the SOURCE plane from the source and check using cross-products if the
        source is inside one of the two triangles.
        """
        # l1=((yray1-yray2)*(ys1-xray2)+(xray2-xray1)*(ys2-yray2))/((yray1-yray2)*(xray-xray2)+(xray2-xray1)*(yray-yray2))

        x1 = y1s - xray
        y1 = y2s - yray

        x2 = y1s - xray1
        y2 = y2s - yray1

        x3 = y1s - xray2
        y3 = y2s - yray2

        x4 = y1s - xray3
        y4 = y2s - yray3

        prod12 = x1 * y2 - x2 * y1
        prod23 = x2 * y3 - x3 * y2
        prod31 = x3 * y1 - x1 * y3
        prod13 = -prod31
        prod34 = x3 * y4 - x4 * y3
        prod41 = x4 * y1 - x1 * y4

        image = np.zeros(xray.shape)
        mask1 = (np.sign(prod12) == np.sign(prod23)) & (np.sign(prod23) == np.sign(prod31))
        mask2 = (np.sign(prod13) == np.sign(prod34)) & (np.sign(prod34) == np.sign(prod41))
        image[mask1] = 1
        image[mask2] = 2

        # In the following, the choices 'image == 1' and 'image == 2' stand for
        # upper and lower triangles (or viceversa).

        # first kind of images (first triangle)
        images1 = np.argwhere(image == 1)
        xi_images_ = images1[:, 1]
        yi_images_ = images1[:, 0]
        xi_images = xi_images_[(xi_images_ > 0) & (yi_images_ > 0)]
        yi_images = yi_images_[(xi_images_ > 0) & (yi_images_ > 0)]

        # compute the weights
        w = np.array([1. / np.sqrt(x1[xi_images, yi_images] ** 2 + y1[xi_images, yi_images] ** 2),
                      1. / np.sqrt(x2[xi_images, yi_images] ** 2 + y2[xi_images, yi_images] ** 2),
                      1. / np.sqrt(x3[xi_images, yi_images] ** 2 + y3[xi_images, yi_images] ** 2)])
        xif1, yif1 = self.refineImagePositions(xi_images, yi_images, w, 1)

        # second kind of images
        images1 = np.argwhere(image == 2)
        xi_images_ = images1[:, 1]
        yi_images_ = images1[:, 0]
        xi_images = xi_images_[(xi_images_ > 0) & (yi_images_ > 0)]
        yi_images = yi_images_[(xi_images_ > 0) & (yi_images_ > 0)]

        # compute the weights
        w = np.array([1. / np.sqrt(x1[xi_images, yi_images] ** 2 + y1[xi_images, yi_images] ** 2),
                      1. / np.sqrt(x3[xi_images, yi_images] ** 2 + y3[xi_images, yi_images] ** 2),
                      1. / np.sqrt(x4[xi_images, yi_images] ** 2 + y4[xi_images, yi_images] ** 2)])
        xif2, yif2 = self.refineImagePositions(xi_images, yi_images, w, 2)

        xi = np.concatenate([xif1, xif2])
        yi = np.concatenate([yif1, yif2])

        #xi, yi = self.refineImages_Newton(xi, yi)# remove duplicates

        mui=self.mu_image(xi,yi)
        xi = (xi - 1 - (len(self.df.thetax)-1) / 2.0) * self.df.pixel_scale
        yi = (yi - 1 - (len(self.df.thetay)-1) / 2.0) * self.df.pixel_scale
        return (xi, yi, mui)

    def find_images_lenstronomy(self):
        """
        Find image positions using the lenstronomy package.

        Returns:
            tuple: Arrays of image x positions, y positions, and magnifications.
        """
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.Util import util
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
        import lenstronomy.Util.simulation_util as sim_util
        from lenstronomy.Data.imaging_data import ImageData

        if self.df == None:
            return (self.ys1, self.ys2, 1.0)

        # define a lenstronomy LensModel using the deflector deflection angles
        numPix = self.df.thetax.shape[0]
        deltaPix = self.df.pixel_scale
        x_grid_interp, y_grid_interp = util.make_grid(numPix, deltaPix)

        lens_model_interp = LensModel(lens_model_list=['INTERPOL'])
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        kwargs_interp = [{'grid_interp_x': x_axes, 'grid_interp_y': y_axes,
                          'f_': self.df.pot,
                          'f_x': self.df.a1,
                          'f_y': self.df.a2}
                         ]
        # set up a grid for the solver
        solver = LensEquationSolver(lens_model_interp)
        kwargs_data = sim_util.data_configure_simple(
            numPix,
            deltaPix,
            center_ra=0.0,
            center_dec=0.0,
            inverse=False,
        )
        pixel_grid = ImageData(**kwargs_data)
        kwargs_solver = {}

        # solve the lens equation to find the image positions
        xi, yi = solver.image_position_from_source(
            self.ys1,
            self.ys2,
            kwargs_interp,
            search_window=np.max(pixel_grid.width),
            min_distance=pixel_grid.pixel_width,
            solver='lenstronomy',
            **kwargs_solver,
        )

        # find the magnification of the images
        mui = lens_model_interp.magnification(xi, yi, kwargs_interp)
        return xi, yi, mui

    def refineImagePositions_old(self, x, y, w, typ):
        """
        Legacy method to refine image positions as weighted mean of triangle vertices.

        Args:
            x, y (array): Pixel coordinates.
            w (array): Weights for vertices.
            typ (int): Triangle type (1 or 2).

        Returns:
            tuple: Refined x and y positions.
        """
        """Image positions are computed as weighted means of the positions
        of the triangle vertices. The weights are the distances between the
        vertices mapped onto the source plane, and the source position."""

        if (typ == 2):
            xp = np.array([x, x + 1, x + 1])
            yp = np.array([y, y, y + 1])
        else:
            xp = np.array([x, x + 1, x])
            yp = np.array([y, y + 1, y + 1])
        xi = np.zeros(x.size)
        yi = np.zeros(y.size)
        for i in range(x.size):
            xi[i] = (xp[:, i] / w[:, i]).sum() / (1. / w[:, i]).sum()
            yi[i] = (yp[:, i] / w[:, i]).sum() / (1. / w[:, i]).sum()
        return (xi, yi)

    def mu_image(self,xi1,xi2):
        """
        Compute the magnification at given image positions.

        Args:
            xi1, xi2 (array): Image positions.

        Returns:
            array: Magnification values.
        """
        mu = map_coordinates((1.0-self.df.ka)**2-self.df.g1**2-self.df.g2**2,
                             [xi2-1, xi1-1], order=1, prefilter=True)
        return(1./mu) # was returning the absolute magnifciation, now returns the signed one

    def refineImages_Hybrid(self, xi, yi, n_iter=100, tol=1e-3):
        """
        Refine image positions using a fast Jacobian-aided solver for positive
        magnification images and residual minimization for saddle-point images.

        Args:
            xi, yi (array): Initial image positions.
            n_iter (int): Maximum number of iterations.
            tol (float): Tolerance for convergence.

        Returns:
            tuple: Refined x and y positions.
        """
        from scipy.optimize import minimize

        a1 = self.df.a1 / self.df.pixel_scale
        a2 = self.df.a2 / self.df.pixel_scale
        # The solver works in pixel coordinates, so the Jacobian must be
        # differentiated with respect to pixel indices, not arcsec.
        da1_dy, da1_dx = np.gradient(a1)
        da2_dy, da2_dx = np.gradient(a2)

        y1s_pix = self.ys1 / self.df.pixel_scale + (len(self.df.thetax) - 1) / 2.0
        y2s_pix = self.ys2 / self.df.pixel_scale + (len(self.df.thetay) - 1) / 2.0

        xi_refined = []
        yi_refined = []

        mu = self.mu_image(xi, yi)

        def sample(arr, xpix, ypix):
            return float(map_coordinates(arr, [[ypix], [xpix]], order=1, mode='nearest')[0])

        def lens_eq(v):
            xpix = float(v[0])
            ypix = float(v[1])
            alpha1 = sample(a1, xpix, ypix)
            alpha2 = sample(a2, xpix, ypix)
            return np.array([xpix - alpha1 - y1s_pix, ypix - alpha2 - y2s_pix], dtype=float)

        def lens_jac(v):
            xpix = float(v[0])
            ypix = float(v[1])
            da1dx = sample(da1_dx, xpix, ypix)
            da1dy = sample(da1_dy, xpix, ypix)
            da2dx = sample(da2_dx, xpix, ypix)
            da2dy = sample(da2_dy, xpix, ypix)
            return np.array(
                [[1.0 - da1dx, -da1dy], [-da2dx, 1.0 - da2dy]],
                dtype=float,
            )

        def residual(pos):
            r = lens_eq(pos)
            return float(r[0] * r[0] + r[1] * r[1])

        bounds = [(0, self.df.a1.shape[1] - 1), (0, self.df.a1.shape[0] - 1)]

        for i in range(len(xi)):
            x0, y0 = float(xi[i]), float(yi[i])
            mu_i = float(mu[i])

            if mu_i > 0.0 and abs(mu_i) > 1e-3:
                sol = root(
                    lens_eq,
                    x0=np.array([x0, y0], dtype=float),
                    jac=lens_jac,
                    method='hybr',
                    options={'xtol': tol, 'maxfev': n_iter},
                )
                if sol.success and np.all(np.isfinite(sol.x)):
                    x_ref, y_ref = float(sol.x[0]), float(sol.x[1])
                else:
                    res = minimize(
                        residual,
                        x0=[x0, y0],
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'ftol': tol, 'maxiter': n_iter},
                    )
                    x_ref, y_ref = float(res.x[0]), float(res.x[1])
            else:
                res = minimize(
                    residual,
                    x0=[x0, y0],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': tol, 'maxiter': n_iter},
                )
                x_ref, y_ref = float(res.x[0]), float(res.x[1])

            # Optional micro-polish with a few fixed-point Newton steps in pixel
            # coordinates. This helps preserve the tighter alignment we had
            # before the faster solver was introduced.
            for _ in range(3):
                r1, r2 = lens_eq([x_ref, y_ref])
                if float(np.hypot(r1, r2)) < tol:
                    break
                J = lens_jac([x_ref, y_ref])
                detJ = float(np.linalg.det(J))
                if not np.isfinite(detJ) or abs(detJ) < 1e-12:
                    break
                step = np.linalg.solve(J, np.array([r1, r2], dtype=float))
                x_ref -= float(step[0])
                y_ref -= float(step[1])

            xi_refined.append(x_ref)
            yi_refined.append(y_ref)

        return np.array(xi_refined), np.array(yi_refined)
