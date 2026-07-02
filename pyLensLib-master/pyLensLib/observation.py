import numpy as np
from astropy.convolution import convolve_fft, convolve
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from scipy.ndimage import map_coordinates
from skimage.transform import resize


class observation(object):
    """
    Class for handling observational effects such as photon noise and PSF convolution.

    Attributes:
        initialized (bool): Whether the observation is initialized.
        zp (float): Zero point magnitude.
        texp (float): Exposure time.
        bkg (float): Background level.
        noise_init (bool): Whether noise is initialized.
        size (float): Size of the image in arcsec.
        npix (int): Number of pixels.
        pixel (float): Pixel scale in arcsec.
        bkg_counts (float): Background counts per pixel.
    """

    def __init__(self, size=100, Npix=100, zp=24.0, texp=2000, bkg=None, bkg_counts_in=False, mlim=None, rap=None, sn=None):
        """
        Initialize an observation object with noise and PSF parameters.

        Args:
            size (float): Size of the image in arcsec.
            Npix (int): Number of pixels.
            zp (float): Zero point magnitude.
            texp (float): Exposure time.
            bkg (float, optional): Background level.
            bkg_counts_in (bool): If True, background is in counts.
            mlim (float, optional): Limiting magnitude for S/N calculation.
            rap (float, optional): Aperture radius for S/N calculation.
            sn (float, optional): Desired signal-to-noise ratio.

        Returns:
            None
        """
        self.initialized = True
        self.zp = zp
        self.texp = texp
        self.bkg = bkg
        self.noise_init = False
        self.size = size
        self.npix = Npix
        self.pixel = self.size / (self.npix-1) # was self.npix
        if bkg != None:
            if bkg_counts_in:
                self.bkg_counts = self.bkg
            else:
                self.bkg_counts = self.mag2counts(self.bkg) * (self.pixel) ** 2
        elif (rap != None) & (sn != None) & (mlim != None):
            # in this case, the background level is computed based on the desired signal-to-noise
            # ratio for a point source of mag mlim in a circular aperture of radius rap
            # The exposure time should be irrelevant in this operating mode
            self.bkg_counts = self.computeBackground(mlim=mlim,rap=rap,sn=sn)
        else:
            self.bkg_counts = 0.0

    def mag2counts(self, mag_in):
        """
        Convert magnitude to counts per second.

        Args:
            mag_in (float): Input magnitude.

        Returns:
            float: Counts per second.
        """
        return 10 ** (-0.4 * (mag_in - self.zp))

    def counts2mag(self, cnt):
        """
        Convert counts per second to magnitude.

        Args:
            cnt (float): Counts per second.

        Returns:
            float: Magnitude.
        """
        return -2.5 * np.log10(cnt) + self.zp

    def makeNoise(self, image, gaussian = False):
        """
        Create a noise map from an input image.

        Args:
            image (ndarray): Input image (assumed to be in e/s).
            gaussian (bool): If True, returns a Gaussian approximation of the Poisson noise.

        Returns:
            ndarray: Noise map.
        """
        if (image.shape[0] != self.npix):
            raise Exception('Size of image does not match with observation set-up')

        if gaussian:
            sigma = np.sqrt(np.abs(image + self.bkg_counts)*self.texp)
            nx, ny = np.shape(image)
            return np.random.randn(nx,ny) * sigma / self.texp
        else:
            return (np.random.poisson((image + self.bkg_counts) * self.texp, image.shape) / self.texp) \
               - (image + self.bkg_counts)

    def convolve_psf(self, image, psf_image, psf_scale):
        """
        Convolve an image with a PSF using FFT convolution and rescaling.

        Args:
            image (ndarray): Input image.
            psf_image (ndarray): PSF image.
            psf_scale (float): PSF pixel scale.

        Returns:
            ndarray: Convolved image.
        """

        theta = np.linspace(-self.size / 2.0, self.size / 2.0, self.npix)
        x, y = np.meshgrid(theta, theta)
        size_psf = psf_image.shape[0] * psf_scale
        x1pix = (x + size_psf / 2.0) / psf_scale
        x2pix = (y + size_psf / 2.0) / psf_scale
        resc_psf_image = map_coordinates(psf_image, [x2pix, x1pix], order=1, prefilter=True, mode='constant', cval=0.0)
        conv_image: object = convolve_fft(image,
                                          resc_psf_image,
                                          normalize_kernel=True,
                                          allow_huge=True)
        return conv_image

    def convolve_psf2(self, image, psf_image, psf_scale):
        """
        Convolve an image with a PSF using FFT convolution and zoom rescaling.

        Args:
            image (ndarray): Input image.
            psf_image (ndarray): PSF image.
            psf_scale (float): PSF pixel scale.

        Returns:
            ndarray: Convolved image.
        """

        scale_factor = psf_scale / self.pixel
        resc_psf_image = zoom(psf_image, scale_factor)
        resc_psf_image /= resc_psf_image.sum()

        conv_image = fftconvolve(image, resc_psf_image, mode='same')
        return conv_image

    def computeBackground(self, mlim, rap, sn):
        """
        Compute background counts to obtain a given S/N in apertures of radius rap for magnitude mlim.

        Args:
            mlim (float): Limiting magnitude.
            rap (float): Aperture radius.
            sn (float): Desired signal-to-noise ratio.

        Returns:
            float: Counts/pixel/s from background.
        """
        counts_source = self.mag2counts(mlim)
        area_ap = np.pi*rap**2 # area of the aperture in sq. arcsec
        npix_ap = area_ap/self.pixel**2 # number of pixels in the aperture

        counts_backg = (counts_source**2*self.texp/sn**2-counts_source) / npix_ap
        return counts_backg