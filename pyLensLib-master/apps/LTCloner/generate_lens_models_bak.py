import argparse
import os
import numpy as np
from arviz import output_file
from scipy.spatial import ConvexHull
from pyLensLib.lenstool import *
from pyLensLib.deflector import deflector
import lenstool
from lenstool.potentials import dpie
from matplotlib.colors import LogNorm, AsinhNorm
from astropy.cosmology import FlatLambdaCDM
from pyLensLib.srccatalog import srccatalog
from astropy.table import Table
import subprocess
from collections import Counter, defaultdict
from pyLensLib.pointsrc import pointsrc

from matplotlib import cm
from matplotlib.colors import ListedColormap

# replace the ProPlot colormap with a Matplotlib approximation
def blended_colormap(names, ratios, subranges=None, N=256, name="blended"):
    """
    Make a ListedColormap by concatenating segments from existing colormaps.
    - names: list of colormap names (e.g., ["Greys_r", "YlOrBr"])
    - ratios: relative lengths of each segment (same length as names)
    - subranges: optional list of (lo, hi) for each segment in [0, 1]
    """
    ratios = list(ratios)
    total = float(sum(ratios))
    counts = [max(1, int(N * r / total)) for r in ratios]
    counts[-1] = max(1, N - sum(counts[:-1]))  # exact N

    colors = []
    for i, (cname, count) in enumerate(zip(names, counts)):
        lo, hi = (0.0, 1.0) if subranges is None or subranges[i] is None else subranges[i]
        cmap = cm.get_cmap(cname)
        samples = np.linspace(lo, hi, count)
        colors.append(cmap(samples))
    return ListedColormap(np.vstack(colors), name=name)

# approximate ProPlot's 'Oranges1', 'Blues1_r', 'Blues6' with Matplotlib defaults

cmap4 = blended_colormap(
    names=["Greys_r", "Oranges", "Blues_r", "Blues"],
    ratios=(1, 3, 5, 10),
    N=256,
    name="SciVisColorUnevenGrey",
)

cmap_greys_gold = blended_colormap(
    names=["Greys_r", "YlOrBr_r"],
    ratios=(1, 3),                # more room for greys, tune to taste
    subranges=[(0.0, 0.5), (0.0, 1.0)],
    N=256,
    name="GreysToGold"
)

cosmic_web = blended_colormap(
    names     = ["Blues_r", "PuBu", "YlOrBr"],
    ratios    = [0.70, 0.15, 0.15],             # long cool range, short warm tip
    subranges = [(0.20, 0.85), (0.25, 0.90), (0.35, 1.00)],
    name="cosmic_web"
)

cosmic_web_c = blended_colormap(
    names     = ["cividis_r", "afmhot_r"],
    ratios    = [0.5, 0.5],
    subranges = [(0.50, 1.00), (0.5, 1.00)],
    name="cosmic_web_c"
)

def compute_area_from_galaxies(gals,test=False):
    positions = []
    for g in gals:
        x = findInBlock(g, 'x_centre')
        y = findInBlock(g, 'y_centre')
        if x is not None and y is not None:
            positions.append([float(x), float(y)])
    if len(positions) < 3:
        raise ValueError("Not enough galaxies to define a convex hull.")
    positions = np.array(positions)  # Convert to numpy array for slicing
    hull = ConvexHull(positions)
    if test:
        print(f"Convex hull vertices: {hull.vertices}")
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        ax.plot(positions[:, 0], positions[:, 1], 'o', color='blue')
        for simplex in hull.simplices:
            ax.plot(positions[simplex, 0], positions[simplex, 1], 'k-')
        ax.set_title("Convex Hull of Galaxy Positions")
        ax.set_xlabel("X (arcsec)")
        ax.set_ylabel("Y (arcsec)")
        ax.set_aspect('equal')
        plt.show()
        return hull.volume  # Area in arcsec^2 for testing
    else:
        # For actual area calculation, we return the area in arcsec^2
        print(f"Convex hull area: {hull.volume} arcsec^2")
    return hull.volume  # Area in arcsec^2

def computeMaxDist(potentiel):
    maxdist = 0.0
    for pot in potentiel:
        x = findInBlock(pot, 'x_centre')
        y = findInBlock(pot, 'y_centre')
        v = findInBlock(pot, 'v_disp')
        mag = findInBlock(pot, 'mag')
        if x is not None and y is not None and v is not None and mag is not None:
            dist = np.sqrt(x ** 2 + y ** 2)
            if dist > maxdist:
                maxdist = dist
    print(f"Maximum distance from center: {maxdist:.2f} arcsec")
    return maxdist

def writeLenstoolPar(filename='output.par', **blocks):
    writeLenstoolBlock(filename, "runmode", blocks['runmode'], append=False)
    writeLenstoolBlock(filename, "grille", blocks['grille'], append=True)
    writeLenstoolBlock(filename, "potentiel", blocks['potentiel'], append=True)
    writeLenstoolBlock(filename, "cline", blocks['cline'], append=True)
    writeLenstoolBlock(filename, "grande", blocks['grande'], append=True)
    writeLenstoolBlock(filename, "cosmologie", blocks['cosmologie'], append=True)
    writeLenstoolBlock(filename, "champ", blocks['champ'], finalize=True, append=True)

def runTestRadialDistribution(gals, r, density, err_density, popt):
    plt.errorbar(r, density, yerr=err_density, fmt='o', label='Measured')
    r_fit = np.linspace(r.min(), r.max(), 300)
    plt.plot(r_fit, projected_NFW(r_fit, *popt), '-', label='NFW Fit')
    plt.xlabel("Radius")
    plt.ylabel("Surface number density")
    plt.legend()
    plt.grid(True)
    plt.show()

def binmags(mags, mag_min, mag_max, nbins, area_out_arcmin2):
    bins = np.linspace(mag_min, mag_max, nbins)
    counts_mag_, edges_ = np.histogram(mags, bins=bins)
    err_counts2_mag_ = np.sqrt(counts_mag_)

    m_centers_ = 0.5 * (edges_[1:] + edges_[:-1])
    binsize = edges_[2] - edges_[1]

    counts_mag_ = counts_mag_ / binsize / area_out_arcmin2
    err_counts_mag_ = err_counts2_mag_ / binsize / area_out_arcmin2
    return m_centers_, counts_mag_, err_counts_mag_

def runTestLuminosityFunction(m_centers_, counts_mag_, err_counts_mag_, popt_s):
    M_fit = np.linspace(17, 30, 300)
    phi_fit = schechter_mag(M_fit, *popt_s)

    fig, ax = plt.subplots(1,1,figsize=(8,5))
    labels_ = ['Input Data', 'Generated Data']
    for m_centers, counts_mag, err_counts_mag, labels in zip(m_centers_, counts_mag_, err_counts_mag_, labels_):
        ax.errorbar(m_centers, counts_mag, yerr=err_counts_mag, fmt='o', label=labels)
    ax.plot(M_fit, phi_fit, 'r-', label='Schechter fit')
    ax.invert_xaxis()
    ax.set_yscale('log')
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Galaxy count per unit mag and area")
    ax.set_title("Luminosity Function with Schechter Fit")
    ax.grid(True)
    ax.legend()
    plt.show()

def runTestScalingRelations(model_gal, mag0, v_ref, rcut_ref, alpha, beta):
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    vdisp = np.array([findInBlock(g, 'v_disp') for g in model_gal])
    mag = np.array([findInBlock(g, 'mag') for g in model_gal])

    m = np.linspace(16,30,100)
    vdisp_fit = v_disp_from_mag(m, alpha, mag_0=mag0, v_disp_0=v_ref)
    rcut_fit = cut_radius_from_mag(m, beta, mag_0=mag0, cut_radius_0=rcut_ref)


    ax[0].scatter(mag, vdisp, s=10, color='blue', alpha=0.5)
    ax[0].plot(m, vdisp_fit, 'r-', label='Fitted v_disp')
    ax[0].set_xlabel("Magnitude")
    ax[0].set_ylabel("Velocity Dispersion (km/s)")
    ax[0].set_title("Velocity Dispersion vs Magnitude")
    ax[0].legend()
    ax[0].grid(True)

    cut_radius = np.array([findInBlock(g, 'cut_radius') for g in model_gal])

    ax[1].scatter(mag, cut_radius, s=10, color='red', alpha=0.5)
    ax[1].plot(m, rcut_fit, 'g-', label='Fitted Cut Radius')
    ax[1].set_xlabel("Magnitude")
    ax[1].set_ylabel("Cut Radius (arcsec)")
    ax[1].set_title("Cut Radius vs Magnitude")
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()

def runTestLenstoolModel(lt, z_lens, cosmologie, outdir, sim_id, fieldsize=100.0, images=None, ax=None, zs_ref=6.0):
    """
    Generate a test plot for the Lenstool model.
    :param lt: lenstool object containing the model
    :param z_lens: redshift of the lens
    :param cosmologie: cosmological parameters block
    :param outdir: output directory
    :param sim_id: simulation id (for numbering the output image)
    :param fieldsize: side-length of the field of view
    :return: nothing... just display the figure and save it to the output directory
    """
    conv, wcs = lt.g_mass(1, 1000, z_lens, zs_ref)
    a1, a2, wcs = lt.g_dpl(1000, zs_ref)

    kwargs_def = {'zl': z_lens, 'zs': zs_ref}
    co = FlatLambdaCDM(H0=cosmologie['H0'], Om0=cosmologie['omegaM'])
    df = deflector(co, angx=a1, angy=a2, **kwargs_def)
    theta = np.linspace(-fieldsize/2, fieldsize/2, a1.shape[0])
    df.setGrid(theta=theta)
    tl = df.tancl()
    rl = df.radcl()
    ctl = df.getCaustics(tl)
    crl = df.getCaustics(rl)

    create_figure = False
    if ax is None:
        # create a new figure and axis if not provided
        create_figure = True
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    np.save("convergence.npy", conv)
    #ax.imshow(conv, origin='lower', norm=LogNorm(vmin=conv.min(), vmax=conv.max()),
    #          extent=[-fieldsize/2, fieldsize/2, -fieldsize/2, fieldsize/2], cmap=cmap4)

    ax.imshow(conv, origin='lower', norm=AsinhNorm(vmax=3.0),
              extent=[-fieldsize/2, fieldsize/2, -fieldsize/2, fieldsize/2], cmap=cmap4)
    #ax.contour(conv, levels=np.linspace(0.8, 2.3, 10), colors='white', linewidths=0.5,
    #                 extent=[-fieldsize/2, fieldsize/2, -fieldsize/2, fieldsize/2])
    for t in tl:
        x, y = df.getCritPoints(t)
        ax.plot(x, y, '-', color='yellow')

    #for c in ctl:
    #    x, y = df.getCausticPoints(c)
    #    ax.plot(x, y, '-', color='red')

    for t in rl:
        x, y = df.getCritPoints(t)
        ax.plot(x, y, '-', color='yellow')

    #for c in crl:
    #    x, y = df.getCausticPoints(c)
    #    ax.plot(x, y, '-', color='red')

    """
    if images is not None:
        # show image data
        i = 1
        for entry in images:
            ax.plot(entry['RA'],entry['DEC'], 'o', markersize=5, color='orange', alpha=0.5)
            ax.text(entry['RA']+0.1, entry['DEC']+0.1, str(entry['ID']), fontsize=8, color='orange', ha='center', va='center')
    """

    ax.set_xlim(-fieldsize/2, fieldsize/2)
    ax.set_ylim(-fieldsize/2, fieldsize/2)
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")
    ax.set_title("Generated Model")

    if create_figure:
        plt.tight_layout()
        fileout = os.path.join(outdir, f"simulated_model_{sim_id+1:03d}.png")
        fig.savefig(fileout,dpi=300)
        plt.show()

def get_v_disp(p): return float(findInBlock(p, 'v_disp') or 0)

def write_lenstool_image_catalog(filename, data, ref_ra=0.0, ref_dec=0.0):
    with open(filename, "w") as f:
        f.write(f"#REFERENCE 3 {ref_ra} {ref_dec}\n")
        for row in data:
            # Each row must have exactly 8 fields: ID RA DEC a b theta z mag
            f.write(" ".join(str(x) for x in row) + "\n")

def plotMassMapWithImages(lt, z_lens, zs_ref, cosmologie, outdir, sim_id, fieldsize,
                          multiple_images, img_maglim):
    """
    Generate a plot with convergence map, critical lines, and multiple images overlaid.

    :param lt: lenstool object containing the model
    :param z_lens: redshift of the lens
    :param zs_ref: reference source redshift
    :param cosmologie: cosmological parameters block
    :param outdir: output directory
    :param sim_id: simulation id (for numbering the output image)
    :param fieldsize: side-length of the field of view
    :param multiple_images: list of dicts with multiple image data (ID, RA, DEC, mag, etc.)
    :param img_maglim: magnitude limit for displaying images
    :return: nothing... just save the figure
    """
    # Generate convergence map and deflection angles
    conv, wcs = lt.g_mass(1, 1000, z_lens, zs_ref)
    a1, a2, wcs = lt.g_dpl(1000, zs_ref)

    # Create deflector and compute critical lines
    kwargs_def = {'zl': z_lens, 'zs': zs_ref}
    co = FlatLambdaCDM(H0=cosmologie['H0'], Om0=cosmologie['omegaM'])
    df = deflector(co, angx=a1, angy=a2, **kwargs_def)
    theta = np.linspace(-fieldsize/2, fieldsize/2, a1.shape[0])
    df.setGrid(theta=theta)
    tl = df.tancl()
    rl = df.radcl()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Plot convergence map
    ax.imshow(conv, origin='lower', norm=AsinhNorm(vmax=3.0),
              extent=[-fieldsize/2, fieldsize/2, -fieldsize/2, fieldsize/2], cmap=cmap4)

    # Plot tangential critical lines
    for t in tl:
        x, y = df.getCritPoints(t)
        ax.plot(x, y, '-', color='yellow', linewidth=1.5, label='Critical lines' if t == tl[0] else '')

    # Plot radial critical lines
    for t in rl:
        x, y = df.getCritPoints(t)
        ax.plot(x, y, '-', color='yellow', linewidth=1.5)

    # Plot multiple images
    if multiple_images:
        # Group images by source ID
        sources = {}
        for img in multiple_images:
            if img['mag'] < img_maglim:
                src_id = img['ID'].split('.')[0]  # Get source ID without multiplicity
                if src_id not in sources:
                    sources[src_id] = []
                sources[src_id].append(img)

        # Plot images grouped by source with different colors
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(sources))))
        for idx, (src_id, imgs) in enumerate(sources.items()):
            color = colors[idx % len(colors)]
            for img in imgs:
                ax.plot(img['RA'], img['DEC'], 'o', color=color, markersize=8,
                       markeredgecolor='white', markeredgewidth=1.5, alpha=0.8)
                ax.text(img['RA']+0.5, img['DEC']+0.5, img['ID'], fontsize=7,
                       color='white', ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

        # Add legend entry for images
        ax.plot([], [], 'o', color='gray', markersize=8, markeredgecolor='white',
               markeredgewidth=1.5, label=f'Multiple images (N={len([i for i in multiple_images if i["mag"] < img_maglim])})')

    ax.set_xlim(-fieldsize/2, fieldsize/2)
    ax.set_ylim(-fieldsize/2, fieldsize/2)
    ax.set_xlabel("X (arcsec)", fontsize=12)
    ax.set_ylabel("Y (arcsec)", fontsize=12)
    ax.set_title(f"Model {sim_id+1:03d}: Convergence Map with Multiple Images", fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')

    plt.tight_layout()
    # Ensure output directory exists and use absolute path
    outdir_abs = os.path.abspath(outdir)
    os.makedirs(outdir_abs, exist_ok=True)
    fileout = os.path.join(outdir_abs, f"simulated_model_{sim_id+1:03d}_with_images.png")
    fig.savefig(fileout, dpi=300, bbox_inches='tight')
    print(f"Saved figure with multiple images: {fileout}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate lens models from a Lenstool .par file")
    parser.add_argument("parfile", help="Input Lenstool .par file")
    parser.add_argument("nmodels", type=int, help="Number of lens models to generate")
    parser.add_argument("--mmin", type=float, default=17.0, help="Minimum magnitude for luminosity function")
    parser.add_argument("--mmax", type=float, default=26.0, help="Maximum magnitude for luminosity function")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--nbinr", type=int, default=5, help="Number of radial bins for distribution measurement")
    parser.add_argument("--fieldsize", type=float, default=100.0, help="Side length of the field of view (arcsec)")
    parser.add_argument("--src_maglim", type=float, default=30.0, help="Magnitude limit for source galaxies (intrinsic)")
    parser.add_argument("--img_maglim", type=float, default=30.0,
                        help="Magnitude limit for images (lensed sources)")
    parser.add_argument("--bcg", type=int, default=1, help="Number of BCG galaxies to be associated with the main potentials")
    parser.add_argument("--bcg_offset", type=float, default=1.0, help="Maximum offset for BCG galaxies from the main potential center")
    parser.add_argument("--bcg_elloffset", type=float, default=0.2,
                        help="Maximum offset for BCG galaxy ellipticity from the main potential ellipticity")
    parser.add_argument("--bcg_paoffset", type=float, default=10.0,
                        help="Maximum offset for BCG galaxy position angle from the main potential pa")
    parser.add_argument("--opening_angle", type=float, default=180.0, help="Force allignment of mainpot potentials with the opening angle")
    parser.add_argument("--scatter", type=float, default=0.5, help="Lognormal scatter for scaling relations")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Tolerance for distances between main potentials compared to input model")
    parser.add_argument("--nullify_ellipticity", action="store_true", help="Set ellipticity of all galaxies to zero")
    parser.add_argument("--output_dir", default="generated_models", help="Directory to save output models")
    parser.add_argument("--lens", action="store_true", help="Run with generation of multiple images for each model")
    parser.add_argument("--test", action="store_true", help="Run in test mode without saving models")
    parser.add_argument("--plotimages", action="store_true", help="Plot multiple images for each generated model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    zs_ref = 6.0  # Reference redshift for lensing calculations

    runmode = readLenstoolBlock(args.parfile, 'runmode')
    grille = readLenstoolBlock(args.parfile, 'grille')
    potentiel = readLenstoolBlock(args.parfile, 'potentiel')
    cline = readLenstoolBlock(args.parfile, 'cline')
    grande = readLenstoolBlock(args.parfile, 'grande')
    champ = readLenstoolBlock(args.parfile, 'champ')
    cosmologie = readLenstoolBlock(args.parfile, 'cosmologie')

    # override champ parameters with command line arguments
    champ['xmax'] = args.fieldsize / 2.0
    champ['xmin'] = -args.fieldsize / 2.0
    champ['ymax'] = args.fieldsize / 2.0
    champ['ymin'] = -args.fieldsize / 2.0

    area_out_arcmin2 = (args.fieldsize / 60.0) ** 2  # Convert field size from arcsec to arcmin²

    # select main potential, galaxies, and gas
    mainpot = selectPotentielByType(potentiel, ptype='main')
    gals = selectPotentielByType(potentiel, ptype='gal')
    gas = selectPotentielByType(potentiel, ptype='gas')

    # Step 1: Estimate area from galaxy distribution
    area_arcsec2 = compute_area_from_galaxies(gals,test=args.test)
    area_arcmin2 = area_arcsec2 / 3600.0
    print(f"Estimated area: {area_arcmin2:.2f} arcmin² from convex hull")
    print(f"------------------")

    # Step 2: Measure radial distribution
    if len(gals) == 0:
        print("No galaxies found in the potential. Cannot measure radial distribution.")
        n0=1.3e-2
        rs=150.0
        print(f"Using default NFW parameters: n0 = {n0:.2e}, rs = {rs:.2f} arcsec")
        alpha = 0.23
        beta = 0.64
        print(f"Using default scaling relations: alpha = {alpha:.2f}, beta = {beta:.2f}")
    else:
        print(f"Number of galaxies found: {len(gals)}")
        _, _, _, _, alpha, beta = measureScalingRelations(gals)
        print (f"Measured scaling relations: alpha = {alpha:.2f}, beta = {beta:.2f}")
        maxdist = computeMaxDist(gals)
        r, density, counts, err_density, err_counts = measureRadialDistribution(gals,rmax=maxdist/np.sqrt(2),nbins=args.nbinr)
        popt, _ = fitNFWtoRadialDistribution(r, density, err_density)
        n0, rs = popt
        print(f"Fitted NFW parameters: n0 = {n0:.2e}, rs = {rs:.2f} arcsec")
    if args.test:
        runTestRadialDistribution(gals, r, density, err_density, popt)


    # Step 3: Fit luminosity function
    print (f"-------------------")
    if len(gals) > 0:
        deltamag = 0.7
        mag = np.array([float(findInBlock(g, 'mag')) for g in gals if findInBlock(g, 'mag') is not None])
        m_centers, counts_mag, err_counts_mag = measureLuminosityFunction(potentiel, mag_min=np.min(mag), mag_max=np.max(mag),
                                                                          nbins=int((np.max(mag)-np.min(mag))/deltamag)+1,
                                                                          area=area_arcmin2)
        popt_s, pcov_s = fitSchechterFunction(m_centers, counts_mag, err_counts_mag)
        phistar, Mstar, alpha_m = popt_s
        print(f"Fitted Schechter parameters: phistar = {phistar:.2e}, Mstar = {Mstar:.2f}, alpha = {alpha_m:.2f}")
    else:
        print("No galaxies with magnitudes found. Cannot fit luminosity function.")
        phistar, Mstar, alpha_m = 8.3, 18.5, -1.15

    # generate models

    for i in range(args.nmodels):
        seed = args.seed + 1000 + i
        # Sample magnitudes from the fitted luminosity function
        mags = sampleMagnitudesFromFittedLF(phistar, Mstar, alpha_m, mag_min=args.mmin, mag_max=args.mmax,
                                            area=area_out_arcmin2, scatter=0.0, seed=seed)

        if args.test:
            m_centers_, counts_mag_, err_counts_mag_ = binmags(mags, args.mmin, args.mmax, int((args.mmax - args.mmin) / deltamag) + 1, area_out_arcmin2)
            runTestLuminosityFunction([m_centers,m_centers_], [counts_mag,counts_mag_], [err_counts_mag, err_counts_mag_], popt_s)

        # Generate new galaxies from sampled mags
        unmatched_gals = [g for g in gals if findInBlock(g, 'mag') is not None]

        template_gal = unmatched_gals[0]
        v_ref = float(findInBlock(template_gal, 'v_disp'))
        rcut_ref = float(findInBlock(template_gal, 'cut_radius'))
        mag0 = float(findInBlock(template_gal, 'mag'))

        #center = (float(findInBlock(mainpot[0], 'x_centre')), float(findInBlock(mainpot[0], 'y_centre')))
        #ellip = float(findInBlock(mainpot[0], 'ellipticite'))
        #angle = float(findInBlock(mainpot[0], 'angle_pos'))

        gal_blocks = []
        for m in mags:
            g_new = template_gal.copy()
            g_new['mag'] = m
            g_new['v_disp'] = v_disp_from_mag(m, alpha, mag_0=mag0, v_disp_0=v_ref)
            g_new['cut_radius'] = cut_radius_from_mag(m, beta, mag_0=mag0, cut_radius_0= rcut_ref)
            gal_blocks.append(g_new)

        gal_blocks[0] = template_gal.copy()  # Ensure the first galaxy is the template

        # Place galaxies on scaling relations and generate randomized smooth model and gas model
        model_smooth, model_gal, model_gas = generateLenstoolModel(mainpot, gal_blocks, gas, rs,
                                                                   rmax=args.fieldsize/2.0 * np.sqrt(2.0),
                                                                   tolerance=args.tolerance, offset=0.0,
                                                                   scatter=args.scatter, useLogNormal=True,
                                                                   opening_angle=args.opening_angle,
                                                                   seed=seed, randomize_all=True)
        # sort galaxies by velocity dispersion
        model_gal = sorted([p for p in model_gal if findInBlock(p, 'v_disp') is not None],
                            key=get_v_disp, reverse=True)

        print (f"Generated model {i+1}/{args.nmodels} with {len(model_gal)} galaxies, ")

        if 0 < args.bcg <= len(mainpot):
            # in model_gal associate the BCG galaxies with the main potentials
            for j in range(args.bcg):
                if j < len(model_gal):
                    mainp = mainpot[j]# % len(mainpot)]
                    model_gal[j]['x_centre'] = findInBlock(mainp, 'x_centre') + np.random.uniform(-0.5, 0.5) * args.bcg_offset
                    model_gal[j]['y_centre'] = findInBlock(mainp, 'y_centre') + np.random.uniform(-0.5, 0.5) * args.bcg_offset
                    model_gal[j]['ellipticite'] = findInBlock(mainp, 'ellipticite') + np.random.uniform(-0.5, 0.5) * args.bcg_elloffset
                    # Clamp ellipticite to [0, 0.99]
                    model_gal[j]['ellipticite'] = min(max(model_gal[j]['ellipticite'], 0), 0.99)
                    model_gal[j]['angle_pos'] = findInBlock(mainp, 'angle_pos') + np.random.uniform(-0.5, 0.5) * args.bcg_paoffset
                else:
                    break

        #model_smooth = [model_smooth[0]]#, model_smooth[1], model_smooth[2]]
        #model_smooth[0]['v_disp'] = 950.0
        #model_gal = [model_gal[kk] for kk in range(400)]
        model_gas =[]

        # save file with x_centre, y_centre, mag, v_disp for all cluster galaxies with mag < 24
        vdisp = np.array([findInBlock(g, 'v_disp') for g in model_gal])
        mag = np.array([findInBlock(g, 'mag') for g in model_gal])
        x = np.array([findInBlock(g, 'x_centre') for g in model_gal])
        y = np.array([findInBlock(g, 'y_centre') for g in model_gal])
        mask = mag < 22.0 #(vdisp > 80) & (mag < 22)
        file_clmemb = os.path.join(args.output_dir, f"simulated_model_{i+1:03d}_clmemb.csv")
        with open(file_clmemb, 'w') as f:
            f.write("x_centre,y_centre,mag,v_disp\n")
            for j in range(len(model_gal)):
                if mask[j]:
                    f.write(f"{x[j]:.2f},{y[j]:.2f},{mag[j]:.2f},{vdisp[j]:.2f}\n")

        lt = lenstool.Lenstool()
        lt.set_cosmology(cosmologie['H0'], cosmologie['omegaM'], cosmologie['omegaX'], cosmologie['wX'])
        lt.set_field([-args.fieldsize / 2, args.fieldsize / 2, -args.fieldsize / 2, args.fieldsize / 2])  # Set the field of view

        if args.nullify_ellipticity:
            print ("WARNING: Nullifying ellipticity of all galaxies")
            for g in model_gal:
                g['ellipticite'] = 0.0

        for g in model_smooth + model_gal + model_gas:
            lt.add_lens(dpie(g['x_centre'], g['y_centre'], g['ellipticite'], g['angle_pos'],
                             g['z_lens'], g['v_disp'], rc=g['core_radius'], rcut=g['cut_radius']))

        lt.set_grid(128, 1)

        if args.test:
            runTestScalingRelations(model_gal, mag0, v_ref, rcut_ref, alpha, beta)
            if not args.lens:
                runTestLenstoolModel(lt, mainpot[0]['z_lens'], cosmologie, args.output_dir, sim_id=i, fieldsize=args.fieldsize)

        new_potentiel = model_smooth + model_gal + model_gas
        runmode['source'] = [1,f"simulated_model_{i+1:03d}_sources.csv"]
        grille['nombre'] = 128
        grille['nlentille'] = len(new_potentiel)
        runmode['reference']=[3,0.0,0.0]
        cline['nplan'] = [1, zs_ref]

        new_par = os.path.join(args.output_dir, f"simulated_model_{i+1:03d}.par")
        blocks = {
            'runmode': runmode,
            'grille': grille,
            'potentiel': new_potentiel,
            'cline': cline,
            'grande': grande,
            'cosmologie': cosmologie,
            'champ': champ
        }
        writeLenstoolPar(new_par, **blocks)
        print(f"Model saved: {new_par}")

        # generate a source catalog
        kwargs_src = {
            'FOV': args.fieldsize,
            'filtern': 'BPZ/HST_ACS_WFC_F606W.res',
            'useband': 'f775w',
            'recal': 'yes',
            'maglim': args.src_maglim,
            'udfdir': '/Users/maxmen3/stiva/HUDF/',
            'homedir': '/Users/maxmen3/CODES/bpz-1.99.3/SED/',
            'seed': args.seed
        }
        cat = srccatalog(**kwargs_src)
        df_src = cat.get_dataframe()
        print (f"Number of sources generated: {len(df_src)}")

        # drop sources with z < z_lens
        z_lens = gals[0]['z_lens']
        df_src = df_src[df_src['zgal'] >= z_lens]

        a1, a2, wcs = lt.g_dpl(1000,zs_ref)
        kwargs_def = {'zl': z_lens, 'zs': zs_ref}
        co = FlatLambdaCDM(H0=cosmologie['H0'], Om0=cosmologie['omegaM'])
        df = deflector(co, angx=a1, angy=a2, **kwargs_def)

        print (f"Created deflector assuming z_lens = {z_lens:.2f} and zs_ref = {zs_ref:.2f}")
        print (f"Assumed cosmology: H0 = {cosmologie['H0']}, omegaM = {cosmologie['omegaM']}, omegaX = {cosmologie['omegaX']}, wX = {cosmologie['wX']}")
        print (f"Field size: {args.fieldsize} arcsec; xmin = {champ['xmin']}, xmax = {champ['xmax']}, ymin = {champ['ymin']}, ymax = {champ['ymax']}")

        theta = np.linspace(champ['xmin'], champ['xmax'], a1.shape[0])
        df.setGrid(theta=theta)
        # check if sources are within the caustics of the deflector
        xs = df_src['x'].values
        ys = df_src['y'].values
        xs_pixel, ys_pixel = df.arcsec2pixel(xs, ys)

        print(f"Number of sources in the field: {len(xs_pixel)}")
        print(f"Source coordinates in arcsec: xs = {xs.min()} to {xs.max()}, ys = {ys.min()} to {ys.max()}")

        # generate source planes
        z_planes, dl_arr_equisp = cat.generateSrcPlanesDL(co, z_lens=z_lens, z_source_max=11.0, n_planes=100)
        z_source = df_src['zgal'].values

        # Vectorized plane index assignment - MUCH faster than Python loop
        plane_indices = np.zeros(len(z_source), dtype=int)
        mask_behind = z_source > z_lens
        if np.any(mask_behind):
            # Compute distances to all planes at once using broadcasting
            z_diff = np.abs(z_source[mask_behind, np.newaxis] - z_planes[np.newaxis, :])
            plane_indices[mask_behind] = np.argmin(z_diff, axis=1)

        #df_src['plane_index'] = plane_indices
        #df_src['plane_z'] = z_planes[plane_indices]
        df_src = df_src.copy()
        df_src.loc[:, 'plane_index'] = plane_indices
        df_src.loc[:, 'plane_z'] = z_planes[plane_indices]

        # process all sources on a given lens plane
        # new dataframe with the same columns as df_src but only sources on the current lens plane
        selected_sources = []

        for z in np.unique(df_src['plane_z']):
            isel = df_src['plane_z'] == z
            print (f"Processing sources for lens plane z = {z:.2f} with {isel.sum()} sources")
            df.change_redshift(z)
            print (f"Deflector redshift changed to {df.zs:.2f}")
            _ = df.multImaCrossSection()
            print (f"Cross-section calculated for z = {df.zs:.2f} sigma={_}")
            # Select sources on the current lens plane
            xs_plane = xs_pixel[isel]
            ys_plane = ys_pixel[isel]
            # check sources inside caustics
            inside_caustics = df.points_in_UU(xs_plane, ys_plane)
            print (f"Number of inside caustics: {inside_caustics.sum()}")
            print ('------------------------')
            # Select only sources inside caustics for this plane
            df_plane = df_src[isel].copy()
            df_plane = df_plane[inside_caustics]
            selected_sources.append(df_plane)

            """
            ax = df.plot_shapely_polygon()
            df_src_tmp = df_plane#pd.concat(selected_sources, ignore_index=True)
            xs_ = df_src_tmp['x'].values
            ys_ = df_src_tmp['y'].values
            xs_pixel_, ys_pixel_ = df.arcsec2pixel(xs_, ys_)
            ax.plot(xs_pixel_, ys_pixel_, 'o', color='red', markersize=5)

            tl = df.tancl()
            rl = df.radcl()
            ctl = df.getCaustics(tl)
            crl = df.getCaustics(rl)

            for t in tl:
                x, y = df.getCritPoints(t)
                x,y = df.arcsec2pixel(x, y)
                ax.plot(x, y, '-', color='grey')

            for c in ctl:
                x, y = df.getCausticPoints(c)
                x, y = df.arcsec2pixel(x, y)
                ax.plot(x, y, '-', color='black')

            for t in rl:
                x, y = df.getCritPoints(t)
                x, y = df.arcsec2pixel(x, y)
                ax.plot(x, y, '-', color='grey')

            for c in crl:
                x, y = df.getCausticPoints(c)
                x, y = df.arcsec2pixel(x, y)
                ax.plot(x, y, '-', color='black')

            for kk in range(len(xs_)):

                kwargs_src = {'ys1': xs_[kk], 'ys2': ys_[kk], 'flux': 1.0, 'zs': z}
                ps = pointsrc(size=args.fieldsize, Npix=a1.shape[0], gl=df, use_lenstronomy=False, refine=False,
                          **kwargs_src)
                xi, yi, mui = ps.find_images()
                xi_pixel, yi_pixel = df.arcsec2pixel(xi, yi)
                ax.plot(xi_pixel, yi_pixel, 'o', color='orange', markersize=5)
            plt.show()
            """
            df.change_redshift(zs_ref)

        # Concatenate all selected sources into a new DataFrame
        df_src = pd.concat(selected_sources, ignore_index=True)
        print(f"Number of sources after caustics check: {len(df_src)}")

        # convert the DataFrame to the table required by Lenstool
        # Use vectorized operations instead of slow iterrows()
        tab = Table(names=['n', 'x', 'y', 'a', 'b', 'theta', 'z', 'mag'], dtype=['str', *['float', ] * 7])

        # Pre-allocate data arrays for much faster table construction
        n_sources = len(df_src)
        ids = [str(i+1) for i in range(n_sources)]
        xs = df_src['x'].values
        ys = df_src['y'].values
        pas = df_src['PA'].values
        zs = df_src['zgal'].values
        mags = df_src['mag'].values

        # Bulk add rows (much faster than individual add_row calls)
        for i in range(n_sources):
            tab.add_row([ids[i], xs[i], ys[i], 1.0, 1.0, pas[i], zs[i], mags[i]])

        tab.meta['iref'], tab.meta['ref_ra'], tab.meta['ref_dec'] = 3, 0.0, 0.0
        lt.get_sources()

        #tab.write(os.path.join(args.output_dir, f"simulated_model_{i + 1:03d}_images.csv"), format='ascii',overwrite=True)
        write_lenstool_image_catalog(os.path.join(args.output_dir, f"simulated_model_{i + 1:03d}_sources.csv"), tab.as_array())

        if args.lens:
            # Run Lenstool to generate images

            os.chdir(args.output_dir)
            """
            executable = '/Users/maxmen3/miniforge3/envs/lenstool_env/bin/lenstool'  # Path to the Lenstool executable
            # Run external code (waits by default unless you set `subprocess.Popen`)
            try:
                result = subprocess.run([executable, f"simulated_model_{i+1:03d}.par", "-n"], check=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
                print("Execution output:")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error during execution:")
                print(e.stderr)
                raise

            # Check if the output file exists
            output_file = "image.all"
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"{output_file} not found after running {executable}")

            # Read the output file (e.g., image.all)
            with open(output_file, "r") as f:
                lines = f.readlines()

            # Example: parse the image catalog (assuming 8-column format after header)
            image_data = []
            for line in lines:
                if line.strip().startswith("#"):
                    continue  # Skip header
                fields = line.strip().split()
                if len(fields) == 8:
                    # ID, RA, DEC, a, b, theta, z, mag
                    image_data.append({
                        "ID": int(fields[0]),
                        "RA": float(fields[1]),
                        "DEC": float(fields[2]),
                        "a": float(fields[3]),
                        "b": float(fields[4]),
                        "theta": float(fields[5]),
                        "z": float(fields[6]),
                        "mag": float(fields[7])
                    })

            # Count occurrences of each ID
            id_counts = Counter(entry["ID"] for entry in image_data)

            # Keep only entries with duplicate IDs
            filtered_image_data = [entry for entry in image_data if id_counts[entry["ID"]] > 1]
            # Create a counter for each original ID
            id_multiplicity = defaultdict(int)

            # change the ID to be ID.i,... with i being the multiplicity of the image of the same source
            for entry in filtered_image_data:
                original_id = entry["ID"]
                id_multiplicity[original_id] += 1
                entry["ID"] = f"{original_id}.{id_multiplicity[original_id]}"


            print(f"Filtered down to {len(filtered_image_data)} multiple-image entries.")

            # save to csv file
            output_csv = f"simulated_model_{i + 1:03d}_images.csv"
            with open(output_csv, "w") as f:
                f.write("ID,RA,DEC,a,b,theta,z,mag\n")
                for entry in filtered_image_data:
                    f.write(f"{entry['ID']},{entry['RA']},{entry['DEC']},{entry['a']},{entry['b']},{entry['theta']},{entry['z']},{entry['mag']}\n")
            """


            #runTestLenstoolModel(lt, gals[0]['z_lens'], cosmologie, args.output_dir, sim_id=i,
            #                     fieldsize=args.fieldsize,images=filtered_image_data,ax=ax)
            if args.test:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                runTestLenstoolModel(lt, gals[0]['z_lens'], cosmologie, args.output_dir, sim_id=i,
                                     fieldsize=args.fieldsize, ax=ax)

            multiple_images = []

            # OPTIMIZED: Pre-extract all data from DataFrame to avoid slow iterrows()
            # and group sources by redshift to minimize expensive change_redshift() calls
            df_src_reset = df_src.reset_index(drop=True)
            source_data = {
                'x': df_src_reset['x'].values,
                'y': df_src_reset['y'].values,
                'zgal': df_src_reset['zgal'].values,
                'mag': df_src_reset['mag'].values
            }
            n_sources = len(df_src_reset)

            # Group sources by redshift to minimize change_redshift calls
            unique_zs = np.unique(source_data['zgal'])
            print(f"Processing {n_sources} sources at {len(unique_zs)} unique redshifts...")

            for z_current in unique_zs:
                # Change redshift once per unique redshift (not once per source!)
                df.change_redshift(z_current)

                # Find all sources at this redshift
                z_mask = source_data['zgal'] == z_current
                indices = np.where(z_mask)[0]

                print(f"  Finding images for {len(indices)} sources at z={z_current:.2f}")

                # Process all sources at this redshift
                for k in indices:
                    kwargs_src = {
                        'ys1': source_data['x'][k],
                        'ys2': source_data['y'][k],
                        'flux': 1.0,
                        'zs': z_current
                    }

                    # Create pointsrc and find images
                    ps = pointsrc(size=args.fieldsize, Npix=a1.shape[0], gl=df,
                                use_lenstronomy=False, refine=True, **kwargs_src)
                    xi, yi, mui = ps.find_images()

                    if len(xi) <= 1:
                        continue

                    # Compute magnitudes
                    mags_ = -2.5 * np.log10(np.abs(mui)) + source_data['mag'][k]

                    # Sort images by magnitude
                    sort_idx = np.argsort(mags_)
                    xi, yi, mui, mags_ = xi[sort_idx], yi[sort_idx], mui[sort_idx], mags_[sort_idx]

                    # Check if multiple images pass magnitude cut
                    imag = mags_ < args.img_maglim

                    if np.sum(imag) > 1:
                        for kk in range(len(xi)):
                            new_ima = {
                                "ID": f"{k+1}.{kk+1}",
                                "RA": xi[kk],
                                "DEC": yi[kk],
                                "a": 1.0,
                                "b": 1.0,
                                "theta": 0.0,
                                "z": z_current,
                                "mag": mags_[kk]
                            }
                            multiple_images.append(new_ima)
            plt.tight_layout()

            plt.show()

            ## Generate figure with multiple images overlaid on mass map
            if args.plotimages:
                plotMassMapWithImages(lt, gals[0]['z_lens'], zs_ref, cosmologie,
                                    args.output_dir, i, args.fieldsize,
                                    multiple_images, args.img_maglim)

            # save to csv file
            print(f'Number of multiple images: {len(multiple_images)}')

            # OPTIMIZED: Write CSV files in bulk instead of line-by-line
            output_csv = f"simulated_model_{i + 1:03d}_allimages.csv"
            if multiple_images:
                # Build all lines at once
                csv_lines = ["ID,RA,DEC,a,b,theta,z,mag\n"]
                csv_lines.extend([
                    f"{entry['ID']},{entry['RA']},{entry['DEC']},{entry['a']},{entry['b']},{entry['theta']},{entry['z']},{entry['mag']}\n"
                    for entry in multiple_images
                ])
                with open(output_csv, "w") as f:
                    f.writelines(csv_lines)
            else:
                # Write empty file with header
                with open(output_csv, "w") as f:
                    f.write("ID,RA,DEC,a,b,theta,z,mag\n")

            # Write filtered images (below mag limit)
            output_csv = f"simulated_model_{i + 1:03d}_images.csv"
            filtered_images = [entry for entry in multiple_images if entry['mag'] < args.img_maglim]
            nsaved = len(filtered_images)

            if filtered_images:
                csv_lines = ["ID,RA,DEC,a,b,theta,z,mag\n"]
                csv_lines.extend([
                    f"{entry['ID']},{entry['RA']},{entry['DEC']},{entry['a']},{entry['b']},{entry['theta']},{entry['z']},{entry['mag']}\n"
                    for entry in filtered_images
                ])
                with open(output_csv, "w") as f:
                    f.writelines(csv_lines)
            else:
                with open(output_csv, "w") as f:
                    f.write("ID,RA,DEC,a,b,theta,z,mag\n")

            print(f'Number of multiple images below mag cut: {nsaved}')
            os.chdir("../../")

        #lt.set_sources(tab,0.0, 0.0)  # Set the source catalog with reference RA/DEC at (0,0)
        #lt.e_lensing()
        #tab_images = lt.get_images()
        #print (f"Number of images generated: {len(tab_images)}")
        #print (tab_images)

        #tab_images.write(os.path.join(args.output_dir, f"simulated_model_{i+1:03d}_images.ecsv"), format='ecsv', overwrite=True)




if __name__ == "__main__":
    main()
