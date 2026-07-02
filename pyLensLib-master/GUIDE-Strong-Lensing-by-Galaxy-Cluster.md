# GUIDE: Strong Lensing by Galaxy Cluster

This is a _very preliminary_ version of the guide, highly tailored for the work that I (@LBasz) am currently doing, so it may be not super-generic, and will probably need a major revision...

It is strongly suggested to read this Guide while following the code in the notebook `Test/GGSLsims.ipynb`, which also contains richer explanations for the small details (the order of the steps may be a bit different though!).

1. Create the **deflector** object `df`:
    - load the *deflection maps*  `(angx, angy)`;
  	- the function `create_deflector()` has as input values: the two deflection maps `angx` and `angy`, the cosmology `co`, the redshift of the lens `zl`, and the redshift of the source `zs`;
	- extract the number of pixels and the field of view: `npix = a1.shape[0]`, `fov_ray = pixels*(npix-1)`. The information on `pixels` (which is the "arcsec/pixel" in the image) is inside the header of the `.fits` file of the deflection maps;
	- inside `create_deflector()` we create the deflector object `df = deflector()`, which is returned (after some manipulations) at the end of the function;
	- using the method `setGrid()` of the deflector, we create a grid covering the source plane, and compute lens maps such as convergence, shear (and deflection angles, that actually we fed as input). The `npix` and `fov` of this grid are the *same* as the ones of the deflection map (see above);
	- adjust the various properties of the deflector according to the actual redshift `zs` of the source, since the quantities in `df` when initialized are computed with a predefined value for the source redshift of `zsnorm = 1`.
2. Compute all the **critics** and the corresponding **caustics**:
	- the (e.g,. tangential) critical line contours are computed by solving numerically the equation for the (tangential) magnification $\lambda_t = 1/\mu_t=1-\kappa-\gamma$, by imposing the magnification to be divergent $\mu_t\to\infty$, so that we have to find the zero-level contours of the function $1-\kappa-\gamma=0$ (see `tancl()`). The corresponding caustic is computed using the lens equation $\boldsymbol{\beta}=\boldsymbol{\theta}-\boldsymbol{\alpha}(\boldsymbol{\theta})$ (see `mapCrit2Cau()`);
	- choose one particular pair of critic-caustic lines indexed by `idcl` (e.g., `idcl = 0` for the main line). When computed, these curves are automatically ordered by decreasing size;
	- generate a point in a random position inside that particular caustic, which represents the position of the simulated source in the source plane (to begin with, start e.g. with the centre of the caustic);
3. Compute the **SED** to be assigned to the source:
	- choose a template of SED (AGN, starbust galaxy, ...);
	- the SED templates are in the rest-frame, therefore we have to shift them along the $\lambda$-axis so to take into account the redshift of the source, in order to obtain the true SED of the source. This can be done using `pyLensLib.sedcompat` (the internal replacement of the old `barak.sed` utilities), that allows us to manipulate the SED. Pay attention to the fact that the redshift stretches the range of wavelengths! So
	$$
	\lambda_i, \lambda_f \to (1+z)\lambda_i, (1+z)\lambda_f $$
	and then
	$$
	 \Delta \lambda_{z} = (1+z)\Delta\lambda_{z=0}  > \Delta\lambda_{z=0}
	$$
	Therefore we are not doing only a shift of the SED along the $\lambda$ axis, but the shifted SED covers a larger range of wavelengths (larger by a factor $1+z$)! Moreover, since *we have to conserve energy*, the area under the redshifted SED has to be equal to the area under the un-redshifted SED, otherwise magnitude calculations will be wrong when comparing SEDs at different $z_s$. Therefore, since the wavelength range has been stretched by a factor $1+z$, then it follows that the height of the SED must decrease, in such a way that the integral gives the _same_ numerical value. So if we plot the original and the redshifted SEDs without normalizing them, they will have a different height. 
	- this SED has an arbitrary normalization, but if we know the magnitude of the source in a given band (e.g., the F606W filter of HST) we can rescale the SED (using `normalise_to_mag()`) in such a way that its integral in that band gives the flux that corresponds to that magnitude. (The magnitude in given band for instance can be taken from a real observed object). Starting from a given magnitude of the source in a given band (e.g., F606W), we extract the value of the flux `F_nu` using the function `mag2flux()` (`pyLensLib.sedcompat` also contains methods that let us compute, starting from the spectrum of the source, the flux and the magnitude in arbitrary bands);
	- once normalized the SED, we can compute fluxes and magnitudes of the source in *all* other bands using the functions `calc_flux()` and `calc_mag()`. At this point we create a *Dataframe* that contains the information on the magnitude and flux of the source in each band. 
	- To later create the simulated source object, we need the numerical fluxes of the source (i.e., the flux in "counts/s") in each band. Therefore we compute the equivalent counts corresponding to that flux using the function `flux2counts()`. To do that, we first need the *zeropoint* for each filter, which is computed inside the latter function. The zeropoint can be extracted (i) from the header (for each different filter) of the image in which we want to insert the simulated lensed source (e.g., taken from HST), or, (ii) alternatively, we can compute it using the functions inside the `acstools` package. Pay attention that these counts are *unlensed*, and we can use them later to renormalize the flux of the source in each band (see below).
4.  Create the **source** object:
	- create an instance `src` of, e.g., the `sersic` or the `pointsrc` class. Use the same `fov` and `npix` of the deflection map. The flux ("counts/s") of the source can be computed starting from his magnitude in given band, the SED, and the chosen filter, as described before;
	- technically, we should recreate a source object (e.g., an instance of the class `sersic`) for every filter that we are considering, since this source object needs as input a flux (in counts), which depends on the chosen filter. However, we can avoid this step, and renormalize the flux of the image (see below).
5. Create the HDU of the simulated source, giving as input HDU the quantity `src.image`. Then, rewrite the header using the function `formatHeader()`.
6. *Cutout* HDU of simulated source:
	- center the cutout on the centre of the chosen `idcl`-th critical line;
7. *Cutout* background image:
	- center the cutout on the centre of the chosen `idcl`-th critical line (same point as above);
	- save for later the size in pixel of the cutout as `npix_cutout = hdu_hst.header['NAXIS1']`, since later we'll build the `observation` object with the same pixel size (this is not mandatory, since there is an interpolation step that intervenes before the convolution, but we want to avoid using a huge grid to perform the convolution, due to the large computational cost of this operation).
8. Project simulated image's HDU onto the background (HST) HDU using the `reproject` package. (In sostanza, dobbiamo riproiettare l'immagine simulata (che ha la stessa pixelscale della mappa di deflessione) sull'header delle immagini HST, in modo tale che poi l'immagine simulata abbia la stessa pixelscale di HST (scritta nell'header). In questo modo, dato che abbiamo preso due cutout (immagine simulata e immagine HST) della stessa dimensione fisica, possiamo sommare queste due immagini, dato che avendo stessa pixelscale le due immagini avranno anche stesse dimensioni in pixel). 
9. Create `observation` object:
	- use `fov = fov_cutout` and `npix = npix_cutout`
10. Perform the **convolution** of simulated image's cutout with PSF:
	- we extracted cutout of the images *before* performing the convolution, since otherwise it would have been a huge waste of computational resources to convolute an almost-empty large image.
11. Sum background cutout and convolved lensed cutout (the flux of the simulated source (i.e., lensed image) is normalized with respect to a filter in a band).
12. Repeat the previous steps (starting from the creation of the source object) for _all other bands_. Actually, if the shape of the source (SED) does not change with the wavelength, we do not need to repeat all the above process in order to simulate the lensed image in the other bands. In particular, we do not have to generate the image from scratch (by creating a new source object with the new flux, i.e., `se = sersic()`, and the associated header, etc.), but instead we can just renormalize the image in the band that we have already done with the flux ratios of the source in the old and the new bands. 
	- At this point we already have the rescaled fluxes in all bands, i.e., once rescaled the SED for a magnitude in a given band, we can compute the magnitudes and fluxes $F_{\nu}$ in all other bands. What we do here is another thing; to compute the flux in "counts/s" from $F_{\nu}$ we need the zeropoint of the filter, and thus in principle we have to create one instance class `sersic` for each other band (where in the field `'flux'` we put $F_{\nu}$ of each band). Instead we can rescale the counts in the other bands using the one we've just computed.
		- Notice that the K-correction has already been integrated in the program, since `pyLensLib.sedcompat` allows us to shift and stretch the SED as a function of the redshift (under the constraint of conserving the total energy);
13. Finally, stack the 3 layers with `np.dstack()` and plot!
