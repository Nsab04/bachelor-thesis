"""
Utilities for reading and handling the outputs of lenstool files
"""
import pandas as pd
import numpy as np
from astropy.coordinates import Angle
from astropy import units as u
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull, QhullError

def isfloat(val):
    """
    Check if a string can be interpreted as a float.

    Args:
        val (str): String to check.

    Returns:
        bool: True if val is a float, False otherwise.
    """
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def findblockend(lines):
    """
    Find the index of the first line containing the string 'end'.

    Args:
        lines (list): List of lines to inspect.

    Returns:
        int: Index of the first line containing 'end'.
    """
    for i in range(len(lines)):
        if 'end' in lines[i]:
            return i

def getRef_RA_DEC(best_par='best.par'):
    """
    Read the best.par file and extract the reference RA and DEC values.

    Args:
        best_par (str, optional): Path to the best.par file (default is 'best.par').

    Returns:
        tuple: (RA_ref, DEC_ref) in degrees.
    """
    try:
        with open(best_par, "r") as file_read:
            lines = file_read.readlines()
        for i in range(len(lines)):
            if 'reference ' in lines[i]:
                value = [s for s in lines[i].split() if isfloat(s) or s.isnumeric()]
                if int(value[0]) == 1:
                    # convert to degrees
                    ra_angle = Angle(value[1], unit=u.hourangle)
                    ra_degrees = ra_angle.degree
                    dec_angle = Angle(value[2], unit=u.deg)
                    dec_degrees = dec_angle.degree
                    # convert to floats
                    return float(ra_degrees), float(dec_degrees)
                elif int(value[0]) == 3:
                    if isinstance(value[1], list) and len(value[1]) > 0:
                        ra_degrees = float(value[1][0])
                    else:
                        ra_degrees = float(value[1])
                    if isinstance(value[2], list) and len(value[2]) > 0:
                        dec_degrees = float(value[2][0])
                    else:
                        dec_degrees = float(value[2])
                    return ra_degrees, dec_degrees
    except FileNotFoundError:
        print("\nThe file doesn't exist!")
    except Exception:
        raise
    return 0.0, 0.0

def getClMembers(best_par='best.par', proftype="81",
                 fields=('x_centre', 'y_centre', 'ellipticite',
                         'angle_pos', 'core_radius', 'core_radius_kpc',
                         'cut_radius', 'cut_radius_kpc', 'v_disp', 'z_lens')):
    """
    Extract information about cluster member galaxies from a lenstool best.par file.

    Args:
        best_par (str, optional): Path to the best.par file.
        proftype (str, optional): Profile type identifier.
        fields (tuple, optional): Fields to extract for each galaxy.

    Returns:
        pandas.DataFrame: DataFrame containing requested galaxy information.
    """
    try:
        with open(best_par, "r") as file_read:
            lines = file_read.readlines()

        gals = []
        fields_out = list(fields)
        fields_out.insert(0,'gal_id')

        for i in range(len(lines)):
            # find where a potentiel block starts
            if 'potentiel ' in lines[i]:
                gal_id = lines[i].split()[1]
                profilen = [s for s in lines[i + 1].split() if isfloat(s) or s.isnumeric()]

                # only process blocks corresponding to profiles of kind proftype
                if profilen[0] == proftype:
                    tmp_arr = list(np.zeros(len(fields)+1))
                    tmp_arr[0] = gal_id
                    istart = i
                    iend = istart + 1 + findblockend(lines[istart + 1:])
                    for k in range(len(fields)):
                        field = fields[k]
                        for j in range(istart, iend + 1):
                            # Check if field name is in line (handles both tabs and spaces)
                            line_parts = lines[j].split()
                            if len(line_parts) >= 2 and line_parts[0] == field:
                                value = [s for s in line_parts[1:] if isfloat(s) or s.isnumeric()]
                                if len(value) > 0:
                                    if isfloat(value[0]):
                                        tmp_arr[k+1] = float(value[0])
                                    else:
                                        tmp_arr[k+1] = int(value[0])
                                break
                    gal_entry = {fields_out[k]: tmp_arr[k] for k in range(len(fields_out))}
                    gals.append(gal_entry)
        df = pd.DataFrame(gals)
    except FileNotFoundError:
        print("\nThe file doesn't exist!")
        df= None
    except Exception:
        raise
    return df

def getLensRedshift(best_par='best.par', proftype="81"):
    """
    Extract the lens redshift from a lenstool best.par file.

    Args:
        best_par (str, optional): Path to the best.par file.
        proftype (str, optional): Profile type identifier.

    Returns:
        float: Lens redshift.
    """
    try:
        with open(best_par, "r") as file_read:
            lines = file_read.readlines()

        for i in range(len(lines)):
            # find where a potentiel block starts
            if 'potentiel ' in lines[i]:
                profilen = [s for s in lines[i + 1].split() if isfloat(s) or s.isnumeric()]

                # only process blocks corresponding to profiles of kind proftype
                if profilen[0] == proftype:
                    istart = i
                    iend = istart + 1 + findblockend(lines[istart + 1:])
                    for j in range(istart, iend + 1):
                        line_parts = lines[j].split()
                        if len(line_parts) >= 2 and line_parts[0] == 'z_lens':
                            value = [s for s in line_parts[1:] if isfloat(s) or s.isnumeric()]
                            if len(value) > 0:
                                return float(value[0])
    except FileNotFoundError:
        print("\nThe file doesn't exist!")
    except Exception:
        raise
    return 0.0

def getFoV(best_par='best.par'):
    """
    Extract the field of view (FOV) limits from a lenstool best.par file.

    Args:
        best_par (str, optional): Path to the best.par file.

    Returns:
        np.ndarray: Array of FOV limits [xmin, xmax, ymin, ymax].
    """
    fields = ['xmin', 'xmax', 'ymin', 'ymax']
    lims = np.zeros(len(fields))
    try:
        with open(best_par, "r") as file_read:
            lines = file_read.readlines()
        # search for the champ block
        for i in range(len(lines)):
            if 'champ' in lines[i]:
                istart = i
                for k in range(len(fields)):
                    for j in range(istart,len(lines)):
                        line_parts = lines[j].split()
                        if len(line_parts) >= 2 and line_parts[0] == fields[k]:
                            value = [s for s in line_parts[1:] if isfloat(s) or s.isnumeric()]
                            if len(value) > 0:
                                lims[k] = float(value[0])
                            break
    except FileNotFoundError:
        print("\nThe file doesn't exist!")
    except Exception:
        raise
    return lims

def getClMembDelimitingPoints(parfile, dmax=80):
    """
    Get the convex hull vertices of cluster members within dmax of the FOV center.

    Args:
        parfile (str): Path to the lenstool parameter file.
        dmax (float): Maximum distance from center in arcsec.

    Returns:
        tuple: (x1, x2) lists of hull vertex coordinates (closed polygon).
               Returns empty lists if fewer than 3 points or hull computation fails.
    """
    clmemb_ = getClMembers(parfile)
    lims = getFoV(parfile)
    xcen = 0.5 * (lims[1] + lims[0])
    ycen = 0.5 * (lims[3] + lims[2])

    clmemb = clmemb_.loc[(np.abs(clmemb_.x_centre.values - xcen) < dmax) &
                        (np.abs(clmemb_.y_centre.values - ycen) < dmax)]

    xp = clmemb.x_centre.values - xcen
    yp = clmemb.y_centre.values - ycen

    # Need at least 3 points for a valid convex hull
    if len(xp) < 3:
        print(f"Warning: Only {len(xp)} cluster members found within dmax={dmax}. "
              "Returning bounding box instead.")
        if len(xp) == 0:
            return [], []
        # Return bounding box for 1-2 points (centered coordinates)
        xmin, xmax = np.min(xp), np.max(xp)
        ymin, ymax = np.min(yp), np.max(yp)
        # Add small padding if points are identical
        if xmin == xmax:
            xmin, xmax = xmin - 1, xmax + 1
        if ymin == ymax:
            ymin, ymax = ymin - 1, ymax + 1
        x1 = [xmin, xmax, xmax, xmin, xmin]
        x2 = [ymin, ymin, ymax, ymax, ymin]
        return x1, x2

    # compute the FOV using convex hull
    points = np.column_stack((xp, yp))

    # Remove duplicate points
    points_unique = np.unique(points, axis=0)

    if len(points_unique) < 3:
        print(f"Warning: Only {len(points_unique)} unique cluster member positions. "
              "Returning bounding box instead.")
        xmin, xmax = np.min(xp), np.max(xp)
        ymin, ymax = np.min(yp), np.max(yp)
        if xmin == xmax:
            xmin, xmax = xmin - 1, xmax + 1
        if ymin == ymax:
            ymin, ymax = ymin - 1, ymax + 1
        x1 = [xmin, xmax, xmax, xmin, xmin]
        x2 = [ymin, ymin, ymax, ymax, ymin]
        return x1, x2

    # Check for collinearity (rank < 2)
    centered = points_unique - points_unique.mean(axis=0)
    rank = np.linalg.matrix_rank(centered)
    if rank < 2:
        print("Warning: Cluster members are collinear. Returning bounding box instead.")
        xmin, xmax = np.min(xp), np.max(xp)
        ymin, ymax = np.min(yp), np.max(yp)
        if xmin == xmax:
            xmin, xmax = xmin - 1, xmax + 1
        if ymin == ymax:
            ymin, ymax = ymin - 1, ymax + 1
        x1 = [xmin, xmax, xmax, xmin, xmin]
        x2 = [ymin, ymin, ymax, ymax, ymin]
        return x1, x2

    try:
        hull = ConvexHull(points_unique)
        # Return centered coordinates (relative to FOV center)
        x1 = list(points_unique[hull.vertices, 0])
        x2 = list(points_unique[hull.vertices, 1])
        x1.append(x1[0])
        x2.append(x2[0])
        return x1, x2
    except QhullError as e:
        print(f"Warning: ConvexHull failed ({e}). Returning bounding box instead.")
        xmin, xmax = np.min(xp), np.max(xp)
        ymin, ymax = np.min(yp), np.max(yp)
        if xmin == xmax:
            xmin, xmax = xmin - 1, xmax + 1
        if ymin == ymax:
            ymin, ymax = ymin - 1, ymax + 1
        x1 = [xmin, xmax, xmax, xmin, xmin]
        x2 = [ymin, ymin, ymax, ymax, ymin]
        return x1, x2

def getCosmo(best_par='best.par'):
    """
    Extract cosmological parameters from a lenstool best.par file and create an astropy.cosmology object.

    Args:
        best_par (str, optional): Path to the best.par file.

    Returns:
        astropy.cosmology.w0waCDM: Cosmology model used by lenstool.
    """
    from astropy.cosmology import w0waCDM
    fields = ['H0', 'omegaM', 'omegaX', 'omegaK', 'wX', 'wa']
    cosvals = np.zeros(len(fields))
    try:
        with open(best_par, "r") as file_read:
            lines = file_read.readlines()
        # search for the champ block
        for i in range(len(lines)):
            if 'cosmologie' in lines[i]:
                istart = i
                for k in range(len(fields)):
                    for j in range(istart,len(lines)):
                        line_parts = lines[j].split()
                        if len(line_parts) >= 2 and line_parts[0] == fields[k]:
                            value = [s for s in line_parts[1:] if isfloat(s) or s.isnumeric()]
                            if len(value) > 0:
                                cosvals[k] = float(value[0])
                            break
        co = w0waCDM(H0=cosvals[0],Om0=cosvals[1],Ode0=cosvals[2],w0=cosvals[4],wa=cosvals[5])
    except FileNotFoundError:
        print("\nThe file doesn't exist!")
        return None
    except Exception:
        raise
    return co

def sampleBayes(bayes_file="bayes.dat", n=100):
    """
    Randomly sample n lines from a lenstool bayes.dat file and write to a new file.

    Args:
        bayes_file (str, optional): Path to the bayes.dat file.
        n (int, optional): Number of lines to sample.

    Returns:
        None
    """
    try:
        lines = []
        with open(bayes_file, "r") as file_read, open(bayes_file + '.random', "w") as file_write:
            for line in file_read.readlines():
                if line.startswith('#'):
                    file_write.write(line)
                else:
                    lines.append(line)
            nindexes = np.random.randint(0,len(lines),n)
            for i in range(n):
                file_write.write(lines[nindexes[i]])
    except FileNotFoundError:
        print("\nThe file doesn't exist!")
    except Exception:
        raise
    return


def create_deflector(parfile, filex=None, filey=None, filepot=None, usePotential=False, zl=0.3, zs=1.0, zsnorm=1.0, resc_fact=1.0, compute_potential=False):
    """
    Create a deflector object from lenstool deflection angle or potential maps.

    Args:
        parfile (str): Lenstool parameter file.
        filex (str, optional): Map of the first component of the deflection angles.
        filey (str, optional): Map of the second component of the deflection angles.
        filepot (str, optional): Map of the lensing potential.
        usePotential (bool, optional): Use potential instead of deflection angles.
        zl (float, optional): Lens redshift.
        zs (float, optional): Source redshift for which the deflector properties are computed.
        zsnorm (float, optional): Source redshift used by lenstool to compute the deflection angle maps.
        resc_fact (float, optional): Rescaling factor for the deflector lensing maps.
        compute_potential (bool, optional): If True, compute the potential map.

    Returns:
        deflector: Deflector object.
    """

    from skimage.transform import rescale
    import astropy.io.fits as pyfits
    from pyLensLib.deflector import deflector

    co = getCosmo(best_par=parfile)

    if not usePotential:
        angx_hd = pyfits.open(filex)
        angy_hd = pyfits.open(filey)
        a1 = angx_hd[0].data
        a2 = angy_hd[0].data
        npix = a1.shape[0]
        if filepot is not None:
            pot = pyfits.open(filepot)[0].data
    else:
        angx_hd = pyfits.open(filepot)
        pot = angx_hd[0].data
        npix = pot.shape[0]

    try:
        if (angx_hd[0].header['CDELT1'] == 1.0):
            pixels = -3600.0 * angx_hd[0].header['CD1_1']
        else:
            pixels = -3600.0 * angx_hd[0].header['CDELT1']
    except KeyError:
        pixels = -3600.0 * angx_hd[0].header['CD1_1']

    if resc_fact != 1.0:
        if not usePotential:
            a1 = rescale(a1, resc_fact, anti_aliasing=False)
            a2 = rescale(a2, resc_fact, anti_aliasing=False)
            if filepot is not None:
                pot = rescale(pot, resc_fact, anti_aliasing=False)
            npix = a1.shape[0]
        else:
            pot = rescale(pot, resc_fact, anti_aliasing=False)
            npix = pot.shape[0]
        pixels = pixels / resc_fact

    fov_ray = pixels * (npix - 1)
    theta = np.linspace(-fov_ray / 2., fov_ray / 2., npix)

    kwargs_def = {'zl': zl, 'zs': zsnorm}

    if not usePotential:
        if filepot is None:
            df = deflector(co, angx=a1, angy=a2, **kwargs_def)
            df.setGrid(theta=theta, compute_potential=compute_potential)
        else:
            df = deflector(co, angx=a1, angy=a2, pot=pot, **kwargs_def)
            df.setGrid(theta=theta, compute_potential=False)
    else:
        df = deflector(co, pot=pot, usePotential=True, **kwargs_def)
        df.setGrid(theta=theta, compute_potential=compute_potential)

    df.change_redshift(zs)
    return df


def read_data_file(file_path):
    """
    Read a lenstool output data file and return its type, reference RA/DEC, and data as a pandas DataFrame.

    Args:
        file_path (str): Path to the data file.

    Returns:
        tuple: (type, ra_ref, dec_ref, data) where data is a pandas DataFrame.
    """
    with open(file_path, 'r') as file:
        # Read the first line
        first_line = file.readline().strip()

        # Check if the first line contains the word "REFERENCE"
        if '#REFERENCE' in first_line:
            parts = first_line.split()
            type_ = int(parts[parts.index('#REFERENCE') + 1])
            ra_ref = float(parts[parts.index('#REFERENCE') + 2])
            dec_ref = float(parts[parts.index('#REFERENCE') + 3])
        else:
            raise ValueError("The first line does not contain the word '#REFERENCE'.")

    try:
        # Read the rest of the file into a pandas DataFrame
        data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
        print(data.head())  # Print the first few lines of the dataframe
        return type_, ra_ref, dec_ref, data
    except pd.errors.ParserError:
        print("Error parsing the file. Please check the file format.")
    except Exception as e:
        print(f"An error occurred: {e}")


def write_data_file(data, file_path):
    """
    Write a pandas DataFrame to a lenstool input data file.

    Args:
        data (pandas.DataFrame): Data to write.
        file_path (str): Path to the output file.

    Returns:
        None
    """
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the first line
        file.write("#REFERENCE 0\n")

        # Write the DataFrame to the file
        data.to_csv(file, sep=' ', index=False, header=True)

def readLenstoolBlock(best_par='best.par', block_name='runmode'):
    """
    Read a named block from a lenstool best.par file and return its contents.

    Args:
        best_par (str, optional): Path to the best.par file.
        block_name (str, optional): Name of the block to read.

    Returns:
        dict or list: Block contents as a dict (single block) or list (multi-block).
    """
    results = []
    current_block = None
    current_id = None
    collecting = False

    with open(best_par, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # Start of a new block
            if stripped.startswith(block_name):
                collecting = True
                current_block = {}
                parts = stripped.split()
                if block_name == "potentiel" and len(parts) > 1:
                    current_block['id'] = parts[1]  # Save the ID
                continue

            # End of block
            if collecting and stripped == 'end':
                results.append(current_block)
                collecting = False
                continue

            # Collect key-value pairs
            if collecting:
                parts = stripped.split()
                key = parts[0]
                values = [float(p) if isfloat(p) else p for p in parts[1:]]
                current_block[key] = values[0] if len(values) == 1 else values

    # Return the full list if it's a multi-block case (like potentiel), else the first one
    if block_name == "potentiel":
        return results
    else:
        return results[0] if results else {}


def writeLenstoolBlock(filename, block_name, block_data, finalize=False, append=False):
    """
    Write a block of data to a lenstool parameter file.

    Args:
        filename (str): Path to the output file.
        block_name (str): Name of the block to write.
        block_data (dict): Data for the block.
        finalize (bool, optional): If True, finalize the file after writing.
        append (bool, optional): If True, append to the file instead of overwriting.

    Returns:
        None
    """
    def clean_value(value):
        def fmt(val):
            if isinstance(val, (int, float)):
                return str(int(val)) if float(val).is_integer() else f"{val:.6f}"
            return str(val)

        if isinstance(value, list):
            if not value:
                return None
            return " ".join(fmt(v) for v in value)
        return fmt(value)

    def format_block(name, data):
        lines = []

        # Block header
        if name == "potentiel" and "id" in data:
            lines.append(f"{name} {data['id']}")
        else:
            lines.append(name)

        for key, value in data.items():
            if name == "potentiel" and key == "id":
                continue
            formatted = clean_value(value)
            if formatted is not None:
                lines.append(f"\t{key}\t{formatted}")  # Force tab between key and value

        lines.append("\tend")
        return "\n".join(lines)

    # Prepare all blocks
    if isinstance(block_data, list):
        formatted_blocks = [format_block(block_name, b) for b in block_data]
    else:
        formatted_blocks = [format_block(block_name, block_data)]

    mode = "a" if append else "w"
    with open(filename, mode) as f:
        for block in formatted_blocks:
            f.write("\n" + block)
        if finalize:
            f.write("\nfini\n")

def findInBlock(block, keyword):
    """
    Search for a keyword in a Lenstool block dictionary.

    Args:
        block (dict): The dictionary representing a block, e.g., from readLenstoolBlock.
        keyword (str): The key to search for.

    Returns:
        The value associated with the keyword, or None if not found.
    """
    return block.get(keyword, None)

def measureScalingRelations(potentiel,fitting=False):
    """
    Measure scaling relations between magnitude, velocity dispersion, cut radius, and core radius for cluster members.

    Args:
        potentiel (list): List of blocks (dicts) containing galaxy properties (mag, v_disp, cut_radius, core_radius).
        fitting (bool, optional): If True, fit power-law relations using all data; if False, use first two entries for analytic calculation.

    Returns:
        tuple:
            - mag (np.ndarray): Array of magnitudes.
            - v_disp (np.ndarray): Array of velocity dispersions.
            - cut_radius (np.ndarray): Array of cut radii.
            - core_radius (np.ndarray): Array of core radii.
            - alpha (float): Slope of the velocity dispersion-magnitude relation.
            - beta (float): Slope of the cut radius-magnitude relation.
    """
    mag = []
    v_disp = []
    cut_radius = []
    core_radius = []
    for pot in potentiel:
        mag_ = findInBlock(pot, 'mag')
        v_disp_ = findInBlock(pot, 'v_disp')
        cut_radius_ = findInBlock(pot, 'cut_radius')
        core_radius_ = findInBlock(pot, 'core_radius')
        if mag_ is not None and v_disp_ is not None and cut_radius_ is not None and core_radius_ is not None:
            print(f"mag: {mag_}, v_disp: {v_disp_}, cut_radius: {cut_radius_}, core_radius: {core_radius_}")
            mag.append(mag_)
            v_disp.append(v_disp_)
            cut_radius.append(cut_radius_)
            core_radius.append(core_radius_)

    mag = np.array(mag)
    v_disp = np.array(v_disp)
    cut_radius = np.array(cut_radius)
    core_radius = np.array(core_radius)
    if fitting:
        alpha = fitAlphaPowerLaw10(mag, v_disp, x_0=mag[0], y_0=v_disp[0])
        beta = fitAlphaPowerLaw10(mag,cut_radius, x_0=mag[0], y_0=cut_radius[0])
    else:
        alpha = -2.5 * np.log10(v_disp[0] / v_disp[1]) / (mag[0] - mag[1])
        beta = -2.5 * np.log10(cut_radius[0] / cut_radius[1]) / (mag[0] - mag[1])
    return mag, v_disp, cut_radius, core_radius, alpha, beta

def v_disp_from_mag(mag, alpha, mag_0=None, v_disp_0=None):
    """
    Calculate the velocity dispersion from magnitude using a power-law relation.

    Args:
        mag (array-like): Magnitudes of the galaxies.
        alpha (float): Slope of the power-law relation.
        mag_0 (float, optional): Reference magnitude. If None, defaults to the first magnitude.
        v_disp_0 (float, optional): Reference velocity dispersion. If None, defaults to the first v_disp.

    Returns:
        v_disp (ndarray): Calculated velocity dispersions.
    """
    if mag_0 is None:
        mag_0 = mag[0]
    if v_disp_0 is None:
        v_disp_0 = 1.0  # Default reference value

    return v_disp_0 * 10 ** (-0.4 * alpha * (mag - mag_0))

def cut_radius_from_mag(mag, beta, mag_0=None, cut_radius_0=None):
    """
    Calculate the cut radius from magnitude using a power-law relation.

    Args:
        mag (array-like): Magnitudes of the galaxies.
        beta (float): Slope of the power-law relation.
        mag_0 (float, optional): Reference magnitude. If None, defaults to the first magnitude.
        cut_radius_0 (float, optional): Reference cut radius. If None, defaults to the first cut_radius.

    Returns:
        cut_radius (ndarray): Calculated cut radii.
    """
    if mag_0 is None:
        mag_0 = mag[0]
    if cut_radius_0 is None:
        cut_radius_0 = 1.0  # Default reference value

    return cut_radius_0 * 10 ** (-0.4 * beta * (mag - mag_0))

def core_radius_from_mag(mag, mag_0=None, core_radius_0=None):
    """
    Calculate the core radius from magnitude using a power-law relation.

    Args:
        mag (array-like): Magnitudes of the galaxies.
        mag_0 (float, optional): Reference magnitude. If None, defaults to the first magnitude.
        core_radius_0 (float, optional): Reference core radius. If None, defaults to the first core_radius.

    Returns:
        core_radius (ndarray): Calculated core radii.
    """
    if mag_0 is None:
        mag_0 = mag[0]
    if core_radius_0 is None:
        core_radius_0 = 1.0  # Default reference value

    return core_radius_0 * 10 ** (-0.2 * (mag - mag_0))

def createPotentielBlock(gal_id, mag, rs=120, zlens=0.4, alpha=0.28, beta=0.64, mag_0=17.0, v_disp_0=300.0,
                         cut_radius_0=10.0, core_radius_0=1e-4, rmax=100.0, seed=42):
    """
    Create a potentiel block dictionary for a galaxy with a given ID and magnitude.

    Args:
        gal_id (str): Identifier for the galaxy.
        mag (float): Magnitude of the galaxy.

    Returns:
        dict: A dictionary representing the potentiel block.
    """
    N=1
    v_disp = v_disp_from_mag(mag, alpha=alpha, mag_0=mag_0, v_disp_0=v_disp_0)
    cut_radius = cut_radius_from_mag(mag, beta=beta, mag_0=mag_0, cut_radius_0=cut_radius_0)
    core_radius = core_radius_from_mag(mag, mag_0=mag_0, core_radius_0=core_radius_0)

    sampled_r = sample_radius_from_projected_nfw(rs=rs, size=N, rmax=rmax, seed=seed + 10)
    sample_theta = np.random.uniform(0, 2 * np.pi, N)
    x_centre = sampled_r * np.cos(sample_theta)
    y_centre = sampled_r * np.sin(sample_theta)
    ellipticite = np.random.uniform(0.0, 0.9, N)  # Random ellipticity
    angle_pos = np.random.uniform(0.0, 2 * np.pi, N)  # Random position angle

    return {
        'potentiel': gal_id,
        'id': gal_id,
        'mag': mag,
        'v_disp': v_disp,
        'cut_radius': cut_radius,
        'core_radius': core_radius,
        'ellipticite': ellipticite,  # Default value, can be modified later
        'angle_pos': angle_pos,  # Default value, can be modified later
        'x_centre': x_centre,  # Default value, can be modified later
        'y_centre': y_centre,   # Default value, can be modified later
        'z_lens': zlens,  # Default lens redshift
    }

def measureRadialDistribution(potentiel, rmin=0.0, rmax=100.0, nbins=10):
    """
    Measure the radial surface number density distribution of galaxies
    in Lenstool potentiel blocks.

    Args:
        potentiel : list of dicts
            List of potentiel blocks (each a dict) from a Lenstool par file.
        rmin : float
            Minimum radius to consider.
        rmax : float
            Maximum radius to consider.
        nbins : int
            Number of radial bins.

    Returns:
        r_centers : ndarray
            Midpoints of radial bins.
        density : ndarray
            Surface number density (galaxies per unit area) in each bin.
        counts : ndarray
            Number of galaxies per bin (before normalization).
    """
    radii = []

    for pot in potentiel:
        x_centre_ = findInBlock(pot, 'x_centre')
        y_centre_ = findInBlock(pot, 'y_centre')
        mag_ = findInBlock(pot, 'mag')  # Optional: only include galaxies with mag
        if x_centre_ is not None and y_centre_ is not None and mag_ is not None:
            r = np.sqrt(float(x_centre_)**2 + float(y_centre_)**2)
            radii.append(r)

    radii = np.array(radii)
    bins = np.linspace(rmin, rmax, nbins + 1)
    counts, edges = np.histogram(radii, bins=bins)

    # Compute annular areas
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)
    density = counts / area
    err_counts = np.sqrt(counts)
    err_density = err_counts / area

    r_centers = 0.5 * (edges[1:] + edges[:-1])

    return r_centers, density, counts, err_density, err_counts


def measureLuminosityFunction(potentiel, mag_min=10.0, mag_max=30.0, nbins=20, area=1.0):
    """
    Measure the luminosity function (number of galaxies per magnitude bin)
    from Lenstool potentiel blocks.

    Args:
        potentiel : list of dicts
            List of potentiel blocks (each a dict) from a Lenstool par file.
        mag_min : float
            Minimum magnitude to consider.
        mag_max : float
            Maximum magnitude to consider.
        nbins : int
            Number of magnitude bins.

    Returns:
        mag_centers : ndarray
            Midpoints of the magnitude bins.
        luminosity_function : ndarray
            Number of galaxies per unit mag and area.
        err_counts : ndarray
            Poisson uncertainties (sqrt(N)) per bin.
    """
    magnitudes = []

    for pot in potentiel:
        mag = findInBlock(pot, 'mag')
        if mag is not None:
            try:
                magnitudes.append(float(mag))
            except ValueError:
                continue

    magnitudes = np.array(magnitudes)
    bins = np.linspace(mag_min, mag_max, nbins + 1)
    counts, edges = np.histogram(magnitudes, bins=bins)
    err_counts = np.sqrt(counts)

    mag_centers = 0.5 * (edges[1:] + edges[:-1])
    counts = counts / (bins[1]-bins[0]) / area
    err_counts = err_counts / area/ (bins[1]-bins[0])

    return mag_centers, counts, err_counts

def fitAlphaPowerLaw10(x, y, x_0=None, y_0=None):
    """
    Fit the relation: y = y_0 * 10^(-0.4 * alpha * (x - x_0))

    Args:
        x : array-like
            Independent variable (e.g. magnitude, luminosity, etc.)
        y : array-like
            Dependent variable (e.g. velocity dispersion, mass, etc.)
        x_0 : float, optional
            Reference x value. If None, defaults to first valid x.
        y_0 : float, optional
            Reference y value. If None, defaults to first valid y.

    Returns:
        alpha : float
            Best-fit slope in the log-linear space
    """
    # Convert and sanitize input
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("Need at least two valid (x, y) pairs for fitting.")

    if x_0 is None:
        x_0 = x[0]
    if y_0 is None:
        y_0 = y[0]

    def model_func(xval, alpha):
        return y_0 * 10 ** (-0.4 * alpha * (xval - x_0))

    popt, _ = curve_fit(model_func, x, y, p0=[1.0])
    return popt[0]


def projected_NFW(R, n0, rs):
    """
    Projected surface density of NFW profile.

    Args:
        R : array-like
            Projected radial distances.
        n0 : float
            Normalization factor.
        rs : float
            Scale radius.

    Returns:
        Sigma(R) : array
            Surface number density at radius R.
    """
    x = np.asarray(R) / rs
    result = np.zeros_like(x)

    # x < 1
    mask1 = x < 1
    if np.any(mask1):
        x1 = x[mask1]
        term1 = np.arccosh(1 / x1)
        result[mask1] = (1 / (x1**2 - 1)) * (1 - (2 / np.sqrt(1 - x1**2)) * np.arctanh(np.sqrt((1 - x1) / (1 + x1))))

    # x == 1
    mask2 = x == 1
    result[mask2] = 1 / 3

    # x > 1
    mask3 = x > 1
    if np.any(mask3):
        x3 = x[mask3]
        result[mask3] = (1 / (x3**2 - 1)) * (1 - (2 / np.sqrt(x3**2 - 1)) * np.arctan(np.sqrt((x3 - 1) / (x3 + 1))))

    return n0 * result

def fitNFWtoRadialDistribution(r, density, err_density, p0=(1.0, 50.0)):
    """
    Fit an NFW surface density profile to measured galaxy density.

    Args:
        r : ndarray
            Radial bin centers.
        density : ndarray
            Measured surface number density.
        err_density : ndarray
            Errors on surface density.
        p0 : tuple
            Initial guess for (n0, rs).

    Returns:
        popt : tuple
            Best-fit parameters (n0, rs).
        pcov : 2D array
            Covariance matrix of the fit.
    """
    mask = (density > 0) & (~np.isnan(density)) & (~np.isnan(err_density)) & (err_density > 0)
    popt, pcov = curve_fit(projected_NFW, r[mask], density[mask], sigma=err_density[mask], p0=p0, absolute_sigma=True)
    return popt, pcov

def schechter_mag(M, phi_star, M_star, alpha):
    """
    Schechter luminosity function in magnitude space.

    Args:
        M : ndarray
            Array of magnitudes.
        phi_star : float
            Normalization.
        M_star : float
            Characteristic magnitude.
        alpha : float
            Faint-end slope.

    Returns:
        ndarray: Number of galaxies per magnitude bin (not normalized).
    """
    x = 10 ** (0.4 * (M_star - M))

    # Handle edge cases to avoid warnings
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    # Only compute where x is positive, finite, and not too large
    # exp(-x) becomes effectively zero for x > 700
    valid = (x > 0) & np.isfinite(x) & (x < 700)

    if np.any(valid):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            x_valid = x[valid]
            power_term = x_valid ** (alpha + 1)
            exp_term = np.exp(-x_valid)
            result_valid = 0.4 * np.log(10) * phi_star * power_term * exp_term

            # Only assign finite results
            finite_mask = np.isfinite(result_valid)
            result[valid] = np.where(finite_mask, result_valid, 0.0)

    return result

def fitSchechterFunction(mag_centers, counts, err_counts=None, p0=None):
    """
    Fit the measured luminosity function with a Schechter function.

    Args:
        mag_centers : ndarray
            Midpoints of magnitude bins.
        counts : ndarray
            Number of galaxies per bin.
        err_counts : ndarray or None
            Uncertainties in counts (optional).
        p0 : list or None
            Initial guess for [phi_star, M_star, alpha].

    Returns:
        popt : list
            Best-fit parameters [phi_star, M_star, alpha].
        pcov : 2D ndarray
            Covariance matrix of the fit.
    """
    mask = counts > 0  # avoid fitting bins with zero count

    if p0 is None:
        p0 = [1e-2, np.median(mag_centers[mask]), -1.0]  # reasonable default guesses

    if err_counts is not None:
        popt, pcov = curve_fit(
            schechter_mag, mag_centers[mask], counts[mask], p0=p0,
            sigma=err_counts[mask], absolute_sigma=True, maxfev=10000
        )
    else:
        popt, pcov = curve_fit(
            schechter_mag, mag_centers[mask], counts[mask], p0=p0,
            maxfev=10000
        )

    return popt, pcov

def sigmaLT2sigma(sigmaLT):
    """
    Convert the velocity dispersion from the Lenstool units to the standard units.
    The conversion is done by multiplying by \sqrt(3/2)

    Args:
        sigmaLT: velocity dispersion in LT units (km/s)

    Returns:
        sigma: velocity dispersion in standard units (km/s)
    """
    return sigmaLT * np.sqrt(3/2)

def addScatterToScalingRelation(x, alpha, y0, x0=None, scatter=0.1, seed=None, useLogNormal=True):
    """
    Generate y = y0 * 10^(-0.4 * alpha * (x - x0)) with optional scatter.

    Args:
        x : array-like
            Independent variable (e.g. magnitude).
        alpha : float
            Scaling slope.
        y0 : float
            Value at reference point x0.
        x0 : float or None
            Reference point; if None, uses x[0].
        scatter : float
            Scatter in log10(y) if log-normal, or fractional if linear.
        seed : int or None
            RNG seed for reproducibility.
        useLogNormal : bool
            Whether to add scatter in log10 space or linearly.

    Returns:
        y_noisy : ndarray
            Scattered values.
    """
    x = np.asarray(x, dtype=float)
    if x0 is None:
        x0 = x[0]

    rng = np.random.default_rng(seed)

    if useLogNormal:
        log_y = np.log10(y0) - 0.4 * alpha * (x - x0)
        log_y += rng.normal(0.0, scatter, size=len(x))
        y_noisy = 10 ** log_y
    else:
        base = y0 * 10 ** (-0.4 * alpha * (x - x0))
        noise = rng.normal(0.0, scatter * base, size=len(x))
        y_noisy = base + noise

    return y_noisy

def plotPotentielPositions(potentiel, scale=0.1, ax=None):
    """
    Plot positions of potentiel components as circles with size ∝ v_disp and color by category:
        - Red: if 'mag' is present
        - Blue: if id starts with 'O'
        - Orange: all other cases

    Args:
        potentiel : list of dict
            List of potentiel blocks (parsed from Lenstool .par).
        scale : float
            Scaling factor for the circle radii (to make them visually appropriate).
        ax : matplotlib axis, optional
            Axis to plot on. If None, creates a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots()

    for pot in potentiel:
        x = findInBlock(pot, 'x_centre')
        y = findInBlock(pot, 'y_centre')
        v = findInBlock(pot, 'v_disp')
        mag = findInBlock(pot, 'mag')
        pid = findInBlock(pot, 'id')

        if x is None or y is None or v is None:
            continue  # skip incomplete components

        # Determine color
        if mag is not None:
            color = 'red'
        elif pid is not None and str(pid).startswith('O'):
            color = 'yellow'
        else:
            color = 'orange'

        # Plot
        xi = float(x)
        yi = float(y)
        vi = float(v)
        circle = plt.Circle((xi, yi), radius=vi * scale, edgecolor=color, facecolor='none', lw=1.5)
        ax.add_patch(circle)
        ax.plot(xi, yi, 'o', markersize=0.5, color=color)

    ax.set_aspect('equal')
    ax.set_xlabel("x_centre")
    ax.set_ylabel("y_centre")
    #ax.set_title("Galaxy positions by category (circle size ∝ v_disp)")
    #ax.grid(True)
    #plt.show()

def getXYfromPotentiel(potentiel):
    """
    Extract x and y coordinates from a list of Lenstool potentiel blocks.

    Args:
        potentiel (list): List of dicts representing potentiel blocks (parsed from Lenstool .par files).

    Returns:
        tuple:
            - x (np.ndarray): Array of x coordinates (arcseconds).
            - y (np.ndarray): Array of y coordinates (arcseconds).
    """
    x = []
    y = []
    for pot in potentiel:
        x_centre_ = findInBlock(pot, 'x_centre')
        y_centre_ = findInBlock(pot, 'y_centre')
        if x_centre_ is not None and y_centre_ is not None:
            x.append(float(x_centre_))
            y.append(float(y_centre_))
    return np.array(x), np.array(y)

def getRADECfromXY(x, y, ra_ref, dec_ref, type=0):
    """
    Convert x, y coordinates (arcseconds) to right ascension (RA) and declination (DEC) in degrees.

    Args:
        x (float or np.ndarray): X coordinate(s) in arcseconds.
        y (float or np.ndarray): Y coordinate(s) in arcseconds.
        ra_ref (float): Reference RA in degrees.
        dec_ref (float): Reference DEC in degrees.
        type (int, optional): 0 for image plane coordinates, 1 for source plane coordinates (default is 0).

    Returns:
        tuple:
            - ra (float or np.ndarray): Right ascension(s) in degrees.
            - dec (float or np.ndarray): Declination(s) in degrees.
    """
    if type == 0:
        ra = ra_ref - x / 3600.0 / np.cos(np.radians(dec_ref))
        dec = dec_ref + y / 3600.0
    else:
        ra = ra_ref + x / 3600.0 / np.cos(np.radians(dec_ref))
        dec = dec_ref - y / 3600.0
    return ra, dec

def selectPotentielByType(potentiel, ptype='main'):
    """
    Select potentiel blocks based on physical type.

    Args:
        potentiel : list of dicts
            All potentiel blocks parsed from the Lenstool par file.
        ptype : str
            One of:
                - 'main'      → blocks with id starting with 'O'
                - 'gal'       → blocks with the keyword 'mag'
                - 'extshear'  → blocks with profil == 14
                - 'gas'       → remaining blocks not matching the above

    Returns:
        selected : list of dicts
            Filtered list of potentiel blocks.
    """
    selected = []

    for pot in potentiel:
        pid = findInBlock(pot, 'id')
        mag = findInBlock(pot, 'mag')
        profil = findInBlock(pot, 'profil')

        if ptype == 'main':
            if pid is not None and str(pid).startswith('O'):
                selected.append(pot)

        elif ptype == 'gal':
            if mag is not None:
                selected.append(pot)

        elif ptype == 'extshear':
            if profil == 14:
                selected.append(pot)

        elif ptype == 'gas':
            is_main = pid is not None and str(pid).startswith('O')
            is_gal = mag is not None
            is_extshear = profil == 14

            if not (is_main or is_gal or is_extshear):
                selected.append(pot)

    return selected

def matchPotentielListsByDistance(list1, list2):
    """
    Match each potentiel in list1 to the closest one in list2 (by projected distance),
    after sorting both by v_disp. Each potentiel in list2 can be matched only once.

    Parameters:
        list1 : list of dict
            First list of potentiel blocks (e.g., primary components).
        list2 : list of dict
            Second list of potentiel blocks to be matched and reordered.

    Returns:
        ordered_list2 : list of dict
            Reordered list2: matched ones first (ordered by list1),
            then unmatched ones.
    """
    def get_v_disp(p): return float(findInBlock(p, 'v_disp') or 0)
    def get_position(p):
        x = findInBlock(p, 'x_centre')
        y = findInBlock(p, 'y_centre')
        if x is None or y is None:
            return np.inf, np.inf
        return float(x), float(y)

    # Step 1: Sort both lists by v_disp (descending)
    sorted1 = sorted([p for p in list1 if findInBlock(p, 'v_disp') is not None],
                     key=get_v_disp, reverse=True)
    sorted2 = sorted([p for p in list2 if findInBlock(p, 'v_disp') is not None],
                     key=get_v_disp, reverse=True)

    matched = []
    used_indices = set()

    # Step 2: For each in sorted1, find closest unused in sorted2
    for p1 in sorted1:
        x1, y1 = get_position(p1)
        min_dist = np.inf
        min_idx = None

        for i, p2 in enumerate(sorted2):
            if i in used_indices:
                continue
            x2, y2 = get_position(p2)
            dist = np.hypot(x2 - x1, y2 - y1)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        if min_idx is not None:
            matched.append(sorted2[min_idx])
            used_indices.add(min_idx)

    # Step 3: Append remaining unmatched potentials from sorted2
    unmatched = [p for i, p in enumerate(sorted2) if i not in used_indices]
    ordered_list2 = matched + unmatched

    return sorted1, ordered_list2


def generateLenstoolModel(mainpot, gals, gaspotentiel, rs, tolerance=0.1, offset=0.0,
                          scatter=0.05, useLogNormal=True, opening_angle=180.0, rmax=75.0,
                          seed=None, randomize_all=False,
                          N_ref=100, r_ref=50,
                          scatter_sigma=None, scatter_rcut=None, rho_sigma_rcut=0.0):
    """
    Generate a randomized Lenstool model from ordered mainpot and their associated galaxies.

    Parameters:
        mainpot : list of dict
            Main halo components
        gals : list of dict
            Galaxies associated to each mainpot component, one-to-one.
        gaspotentiel : list of dict
            Gas components to include.
        rs : float
            scale radius of the projected NFW representing the radial distribution of galaxies.
        tolerance : float
            Allowed variation in relative distances (default 10%).
        offset : float
            Random offset applied to the main halo center.
        useLogNormal : bool
            True for log-normal scatter in v_disp/cut_radius, False for linear.
        scatter_sigma : float or None
            Scatter to apply to sigma relation. If None, falls back to `scatter`.
        scatter_rcut : float or None
            Scatter to apply to rcut relation. If None, falls back to `scatter`.
        rho_sigma_rcut : float
            Correlation coefficient between sigma and rcut residuals.
        seed : int or None
            Random seed for reproducibility.
        randomize_all : bool
            If True, randomize all galaxies, not just those associated with mainpot.
        N_ref : int
            Reference number of galaxies for randomization when randomize_all is True.
        r_ref : float
            Reference radius for randomization when randomize_all is True.

    Returns:
        list of dict : randomized model (main + galaxies + gas).
    """

    mainpot_ordered, gals_ordered = matchPotentielListsByDistance(mainpot, gals)

    rng = np.random.default_rng(seed)

    # Anchor model component (first in mainpot_ordered)
    anchor = mainpot_ordered[0]
    x0, y0 = float(findInBlock(anchor, 'x_centre')), float(findInBlock(anchor, 'y_centre'))
    x_center, y_center = x0, y0

    # Initialize shifts list
    shifts = []

    # Randomize main components
    randomized_main = []
    for i, p in enumerate(mainpot_ordered):
        p_new = copy.deepcopy(p)
        x, y = float(findInBlock(p, 'x_centre')), float(findInBlock(p, 'y_centre'))

        if i == 0:
            # First component is the anchor, can be shifted within +/- offest
            dx = rng.uniform(-offset, offset)
            dy = rng.uniform(-offset, offset)
        else:
            # other components are shifted randomly within a circle around the anchor
            r = np.hypot(x - x0, y - y0) # distance from anchor
            if opening_angle == 180.0:
                # Full circle
                angle = rng.uniform(0.0, 2.0 * np.pi)
            else:
                # Sector of a circle defined by opening_angle
                # Choose between the two symmetric angular cones around major axis
                angle_pos = np.deg2rad(findInBlock(randomized_main[0], 'angle_pos'))  # Position angle in radians
                if rng.random() < 0.5:
                    angle = rng.uniform(angle_pos - np.deg2rad(opening_angle) / 2, angle_pos + np.deg2rad(opening_angle) / 2)
                else:
                    angle = rng.uniform(angle_pos - np.deg2rad(opening_angle) / 2 + np.pi, angle_pos + np.deg2rad(opening_angle) / 2 + np.pi)

            #angle = rng.uniform(0.0,2.0*np.pi)  # Random angle in radians
            r_new = r * rng.uniform(1 - tolerance, 1 + tolerance) # new distance within tolerance
            x_target = x_center + r_new * np.cos(angle) # x coordinate
            y_target = y_center + r_new * np.sin(angle) # y coordinate
            dx = x_target - x # shift along x
            dy = y_target - y # shift along y

        # Store shift for applying to associated galaxies
        shifts.append((dx, dy))

        # Apply shift and randomize shape
        p_new['x_centre'] = x + dx
        p_new['y_centre'] = y + dy
        p_new['ellipticite'] = rng.uniform(0.0, 0.9)
        p_new['angle_pos'] = rng.uniform(-180, 180)
        randomized_main.append(p_new)

    # Apply corresponding shifts to associated galaxies
    randomized_gal_main = []
    for i, g in enumerate(gals_ordered[:len(shifts)]):
        dx, dy = shifts[i]
        g_new = copy.deepcopy(g)
        xg = float(findInBlock(g, 'x_centre'))
        yg = float(findInBlock(g, 'y_centre'))
        g_new['id'] = str(i)
        g_new['x_centre'] = xg + dx
        g_new['y_centre'] = yg + dy
        g_new['ellipticite'] = rng.uniform(0.0, 0.9)
        g_new['angle_pos'] = rng.uniform(-180, 180)
        _ = g_new.pop('core_radius_kpc', None)
        _ = g_new.pop('cut_radius_kpc', None)
        randomized_gal_main.append(g_new)

    # randomize the remaining galaxies
    center = (float(findInBlock(randomized_main[0], 'x_centre')),float(findInBlock(randomized_main[0], 'y_centre')))
    ellip = float(findInBlock(randomized_main[0], 'ellipticite'))
    angle_pos = float(findInBlock(randomized_main[0], 'angle_pos'))
    v_ref = float(findInBlock(randomized_gal_main[0], 'v_disp'))
    r_cut_ref = float(findInBlock(randomized_gal_main[0], 'cut_radius'))
    r_core_ref = float(findInBlock(randomized_gal_main[0], 'core_radius'))
    mag0 = float(findInBlock(randomized_gal_main[0], 'mag'))

    seed_assign = seed + 13 if seed is not None else None
    if randomize_all:
        # If randomize_all is True, use all galaxies for randomization
        randomized_gal_othergal = assignGalaxiesFromNFW(gals, rs, center, ellip, angle_pos,
                                                        v_ref, r_cut_ref, r_core_ref, mag0,
                                                        scatter=scatter, useLogNormal=useLogNormal, rmax=rmax,
                                                        scatter_sigma=scatter_sigma, scatter_rcut=scatter_rcut,
                                                        rho_sigma_rcut=rho_sigma_rcut,
                                                        seed=seed_assign, N_ref=N_ref, r_ref=r_ref)
        randomized_gal = randomized_gal_othergal
    else:
        randomized_gal_othergal = assignGalaxiesFromNFW(gals[len(mainpot_ordered) - 1:], rs, center, ellip, angle_pos,
                                                        v_ref, r_cut_ref, r_core_ref, mag0,
                                                        scatter=scatter, useLogNormal=useLogNormal, rmax=rmax,
                                                        scatter_sigma=scatter_sigma, scatter_rcut=scatter_rcut,
                                                        rho_sigma_rcut=rho_sigma_rcut,
                                                        seed=seed_assign, N_ref=N_ref, r_ref=r_ref)
        randomized_gal = randomized_gal_main + randomized_gal_othergal

    # Randomize gas components
    randomized_gas = []
    for p in gaspotentiel:
        p_new = copy.deepcopy(p)
        x, y = float(findInBlock(p, 'x_centre')), float(findInBlock(p, 'y_centre'))
        r = np.hypot(x - x0, y - y0)
        if opening_angle == 180.0:
            # Full circle
            angle = rng.uniform(0.0, 2.0 * np.pi)
        else:
            # Sector of a circle defined by opening_angle
            # Choose between the two symmetric angular cones around major axis
            angle_pos = np.deg2rad(findInBlock(randomized_main[0], 'angle_pos'))  # Position angle in radians
            if rng.random() < 0.5:
                angle = rng.uniform(angle_pos - np.deg2rad(opening_angle) / 2,
                                    angle_pos + np.deg2rad(opening_angle) / 2)
            else:
                angle = rng.uniform(angle_pos - np.deg2rad(opening_angle) / 2 + np.pi,
                                    angle_pos + np.deg2rad(opening_angle) / 2 + np.pi)
        #angle = rng.uniform(0.0,2.0*np.pi)#np.arctan2(y - y0, x - x0)
        r_new = r * rng.uniform(1 - tolerance, 1 + tolerance)
        x_target = x_center + r_new * np.cos(angle)
        y_target = y_center + r_new * np.sin(angle)
        p_new['x_centre'] = x_target
        p_new['y_centre'] = y_target
        p_new['ellipticite'] = rng.uniform(0.0, 0.9)
        p_new['angle_pos'] = rng.uniform(-180, 180)
        randomized_gas.append(p_new)

    return randomized_main, randomized_gal, randomized_gas

def assignGalaxiesFromNFW(unmatched_gals, rs, main_center, main_ellipticite, main_angle,
                          v_disp_ref, cut_radius_ref, core_radius_ref, mag0, rmax=75.0, scatter=0.1,
                          seed=None, useLogNormal=True,
                          N_ref=100, r_ref=50.0,
                          scatter_sigma=None, scatter_rcut=None, rho_sigma_rcut=0.0):
    """
    Assign positions to unmatched galaxies based on a projected NFW profile.

    Parameters:
        unmatched_gals : list of dicts
            Galaxies not matched to main halos.
        n0, rs : float
            Fitted NFW parameters (surface density).
        main_center : tuple
            (x0, y0) coordinates of the lens center.
        main_ellipticite : float
            Ellipticity of the main halo.
        main_angle : float
            Position angle (degrees) of the main halo.
        v_disp_ref : float or array-like
            v_disp values before scatter.
        cut_radius_ref : float or array-like
            cut_radius values before scatter.
        scatter : float
            Log-normal scatter (in dex).
        scatter_sigma : float or None
            Scatter for sigma relation (dex if useLogNormal=True, fractional otherwise).
            If None, defaults to `scatter`.
        scatter_rcut : float or None
            Scatter for rcut relation (dex if useLogNormal=True, fractional otherwise).
            If None, defaults to `scatter`.
        rho_sigma_rcut : float
            Correlation coefficient between sigma and rcut residuals.
        seed : int or None
            RNG seed for reproducibility.
        N_ref : int
            Reference number of galaxies within r_ref.
        r_ref : float
            Reference radius for fixed density sampling.

    Returns:
        randomized_gals : list of dicts
            Galaxies with updated x/y/v_disp/cut_radius and randomized positions.
    """
    rng = np.random.default_rng(seed)
    x0, y0 = main_center
    angle_rad = np.deg2rad(main_angle)

    N = len(unmatched_gals)
    if N == 0:
        return []
    seed_sample = seed + 10 if seed is not None else None
    sampled_r = sample_radius_from_projected_nfw_anchored(
        rs=rs, size=N, rmax=rmax,
        N_ref=min(N_ref, N), r_ref=r_ref,
        seed=seed_sample
    )
    #sampled_r = sample_radius(N)
    #sampled_r = sample_radius_from_projected_nfw(rs=rs, size=N, rmax=rmax, seed=seed+10)
    #sampled_r = sample_radius_from_projected_nfw_fixed_density(rs=rs, rmax=rmax, N_ref=N_ref, r_ref=r_ref, seed=seed+10)
    sampled_theta = rng.uniform(0, 2*np.pi, N)

    # Convert polar to elliptical (deformed) Cartesian coordinates

    #axis_ratio = (1 - main_ellipticite) / (1 + main_ellipticite)
    randomized_gals = []

    # Extract magnitudes
    try:
        mag_array = np.array([float(findInBlock(g, 'mag')) for g in unmatched_gals])
    except (TypeError, ValueError):
        raise ValueError("Some galaxies are missing the 'mag' keyword required for scaling.")

    # Scatter v_disp and cut_radius using the scaling relation
    mag1 = mag_array[0]
    v_disp1 = float(unmatched_gals[0]['v_disp'])
    cut_radius1 = float(unmatched_gals[0]['cut_radius'])
    denom = (mag0 - mag1)
    if np.isclose(denom, 0.0):
        idx = np.where(~np.isclose(mag_array, mag0))[0]
        if len(idx) > 0:
            j = int(idx[0])
            mag1 = mag_array[j]
            v_disp1 = float(unmatched_gals[j]['v_disp'])
            cut_radius1 = float(unmatched_gals[j]['cut_radius'])
            denom = (mag0 - mag1)

    if np.isclose(denom, 0.0) or v_disp1 <= 0 or cut_radius1 <= 0 or v_disp_ref <= 0 or cut_radius_ref <= 0:
        alpha = 0.23
        beta = 0.64
    else:
        alpha = -2.5 * np.log10(v_disp_ref / v_disp1) / denom
        beta = -2.5 * np.log10(cut_radius_ref / cut_radius1) / denom

    # Correlated scatter between sigma and rcut.
    # Defaults preserve previous behavior when scatter_sigma/scatter_rcut are not provided.
    scatter_sigma_eff = float(scatter if scatter_sigma is None else scatter_sigma)
    scatter_rcut_eff = float(scatter if scatter_rcut is None else scatter_rcut)
    rho_eff = float(np.clip(rho_sigma_rcut, -0.999, 0.999))

    sigma_mean = v_disp_from_mag(mag_array, alpha=alpha, mag_0=mag0, v_disp_0=v_disp_ref)
    rcut_mean = cut_radius_from_mag(mag_array, beta=beta, mag_0=mag0, cut_radius_0=cut_radius_ref)

    if useLogNormal:
        log_sigma_mean = np.log10(np.clip(sigma_mean, 1e-30, None))
        log_rcut_mean = np.log10(np.clip(rcut_mean, 1e-30, None))

        if scatter_sigma_eff > 0 and scatter_rcut_eff > 0 and abs(rho_eff) > 0:
            cov = np.array([
                [scatter_sigma_eff ** 2, rho_eff * scatter_sigma_eff * scatter_rcut_eff],
                [rho_eff * scatter_sigma_eff * scatter_rcut_eff, scatter_rcut_eff ** 2],
            ], dtype=float)
            eps = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=len(mag_array))
            eps_sigma = eps[:, 0]
            eps_rcut = eps[:, 1]
        else:
            eps_sigma = rng.normal(0.0, max(scatter_sigma_eff, 0.0), size=len(mag_array))
            eps_rcut = rng.normal(0.0, max(scatter_rcut_eff, 0.0), size=len(mag_array))

        v_disp_noisy = 10 ** (log_sigma_mean + eps_sigma)
        cut_radius_noisy = 10 ** (log_rcut_mean + eps_rcut)
    else:
        # Linear mode: scatter is fractional on the mean values.
        if scatter_sigma_eff > 0 and scatter_rcut_eff > 0 and abs(rho_eff) > 0:
            cov = np.array([
                [scatter_sigma_eff ** 2, rho_eff * scatter_sigma_eff * scatter_rcut_eff],
                [rho_eff * scatter_sigma_eff * scatter_rcut_eff, scatter_rcut_eff ** 2],
            ], dtype=float)
            frac = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=len(mag_array))
            frac_sigma = frac[:, 0]
            frac_rcut = frac[:, 1]
        else:
            frac_sigma = rng.normal(0.0, max(scatter_sigma_eff, 0.0), size=len(mag_array))
            frac_rcut = rng.normal(0.0, max(scatter_rcut_eff, 0.0), size=len(mag_array))

        v_disp_noisy = sigma_mean * (1.0 + frac_sigma)
        cut_radius_noisy = rcut_mean * (1.0 + frac_rcut)
        v_disp_noisy = np.clip(v_disp_noisy, 1e-12, None)
        cut_radius_noisy = np.clip(cut_radius_noisy, 1e-12, None)

    core_radius = core_radius_from_mag(mag_array, mag_0=mag0, core_radius_0=core_radius_ref)

    #axis_ratio = (1 - main_ellipticite/2.) / (1 + main_ellipticite/2.)
    axis_ratio = (1 - main_ellipticite ) / (1 + main_ellipticite )

    for i, gal in enumerate(unmatched_gals):
        gal_new = copy.deepcopy(gal)

        # Step 1: uniform angle and radius
        r = sampled_r[i]
        theta = sampled_theta[i]
        #print (f"Galaxy {i}: r = {r}, theta = {theta} (angle in radians: {theta})")
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Step 2: apply ellipticity (stretch/compress)
        x_e = x
        y_e = y #* axis_ratio

        # Step 3: rotate by angle_pos
        x_rot = x_e * np.cos(angle_rad) - y_e * np.sin(angle_rad)
        y_rot = x_e * np.sin(angle_rad) + y_e * np.cos(angle_rad)

        gal_new['id'] = str(i + 1000)  # ensure unique IDs
        gal_new['x_centre'] = x0 + x_rot
        gal_new['y_centre'] = y0 + y_rot
        gal_new['v_disp'] = v_disp_noisy[i]
        gal_new['cut_radius'] = cut_radius_noisy[i]
        gal_new['core_radius'] = core_radius[i]
        gal_new['ellipticite'] = np.clip(main_ellipticite + rng.normal(0, 0.05), 0, 0.6)
        gal_new['angle_pos'] = rng.uniform(-180, 180)
        _ = gal_new.pop('core_radius_kpc', None)
        _ = gal_new.pop('cut_radius_kpc', None)

        randomized_gals.append(gal_new)

    return randomized_gals

def _f_proj_nfw(x):
    """
    Dimensionless cumulative mass function for projected NFW profile.

    Parameters:
        x : array-like
            Radius in units of scale radius (R / rs)

    Returns:
        f(x) : array-like
            Cumulative mass profile (unnormalized)
    """
    x = np.asarray(x)
    f = np.zeros_like(x)

    # Avoid numerical issues
    eps = 1e-6
    x_safe = np.clip(x, eps, None)

    # x < 1
    mask1 = x_safe < 1
    x1 = x_safe[mask1]
    if np.any(mask1):
        f[mask1] = np.log(x1 / 2) + (1 / np.sqrt(1 - x1**2)) * np.arccosh(1 / x1)

    # x == 1
    mask2 = np.isclose(x_safe, 1)
    if np.any(mask2):
        f[mask2] = 1 + np.log(0.5)

    # x > 1
    mask3 = x_safe > 1
    x3 = x_safe[mask3]
    if np.any(mask3):
        f[mask3] = np.log(x3 / 2) + (1 / np.sqrt(x3**2 - 1)) * np.arccos(1 / x3)

    return f

def sample_radius_from_projected_nfw(rs, size, rmax=5.0, rmin=0.05, ngrid=5000, seed=None):
    """
    Sample projected radii from an NFW profile using the analytical cumulative mass profile.

    CORRECT IMPLEMENTATION (per user insight):
    - Normalize CDF to a large fixed radius (not rmax)
    - Sample uniformly from [0, CDF(rmax)] instead of [0, 1]
    - This ensures constant number density when rmax changes

    Parameters:
        rs : float
            Scale radius (same units as output) - from observations, fixed
        size : int
            Number of samples
        rmax : float
            Maximum radius to sample (fieldsize/2 × √2)
        rmin : float
            Minimum radius to avoid center pile-up (default: 0.05)
        ngrid : int
            Grid resolution for inverse sampling
        seed : int or None
            Random seed

    Returns:
        r_samples : ndarray
            Sampled projected radii from the NFW profile
    """
    rng = np.random.default_rng(seed)

    # CRITICAL FIX: Normalize to a FIXED large radius (based only on rs, NOT on rmax!)
    # This ensures the CDF shape is perfectly consistent regardless of rmax value
    r_norm = 100.0 * rs  # Fixed reference: typically 3000-15000 arcsec

    # Safety check: if rmax is somehow larger than r_norm, extend the grid
    if rmax > r_norm:
        r_norm = 2.0 * rmax
        print(f"Warning: rmax ({rmax:.1f}) > 100*rs. Extended r_norm to {r_norm:.1f}")

    r_vals = np.logspace(np.log10(rmin), np.log10(r_norm), ngrid)
    x_vals = r_vals / rs
    cdf_vals = _f_proj_nfw(x_vals)
    cdf_vals /= cdf_vals[-1]  # Normalize to 1 at r_norm (FIXED normalization)

    # Find CDF value at rmax (the actual truncation point)
    cdf_at_rmax = np.interp(rmax, r_vals, cdf_vals)

    # KEY INSIGHT: Sample uniformly from [0, CDF(rmax)] instead of [0, 1]
    # This maintains constant density in inner regions
    u = rng.uniform(0, cdf_at_rmax, size)
    r_samples = np.interp(u, cdf_vals, r_vals)

    return r_samples

import numpy as np

def sample_radius_from_projected_nfw_anchored_(
    rs, size, rmax,
    N_ref, r_ref,
    rmin=0.05, ngrid=5000, seed=None
):
    """
    Sample exactly `size` projected radii from an NFW profile truncated at rmax,
    while enforcing exactly `N_ref` radii within r_ref.

    This is the right tool when:
      - size is already determined externally (e.g. size = len(mag))
      - you want the inner distribution (<= r_ref) to stay unchanged when rmax grows
      - extra objects (size - N_ref) should be placed only at radii > r_ref
    """
    if r_ref > rmax:
        raise ValueError(f"r_ref ({r_ref}) must be <= rmax ({rmax}).")
    if N_ref > size:
        raise ValueError(f"N_ref ({N_ref}) must be <= size ({size}).")

    rng = np.random.default_rng(seed)

    # Build a monotonic cumulative F(r) on [rmin, rmax]
    r_vals = np.logspace(np.log10(rmin), np.log10(rmax), ngrid)
    F_vals = _f_proj_nfw(r_vals / rs)  # UNNORMALIZED cumulative (monotonic)

    F_rref = np.interp(r_ref, r_vals, F_vals)
    F_rmax = F_vals[-1]

    # Sample exactly N_ref inside r_ref: u in [0, F(r_ref)]
    u_in = rng.uniform(0.0, F_rref, N_ref)
    r_in = np.interp(u_in, F_vals, r_vals)

    # Sample remaining in the annulus (r_ref, rmax]: u in [F(r_ref), F(rmax)]
    n_out = size - N_ref
    u_out = rng.uniform(F_rref, F_rmax, n_out)
    r_out = np.interp(u_out, F_vals, r_vals)

    r = np.concatenate([r_in, r_out])
    rng.shuffle(r)
    return r

def sample_radius_from_projected_nfw_anchored(
    rs, size, rmax, r_ref, N_ref,
    rmin=0.05, ngrid=5000, seed=None
):
    """
    Draw `size` radii from projected NFW on [rmin, rmax],
    conditioned on exactly `N_ref` of them being < r_ref.

    Every draw is from the NFW distribution (conditional on region).
    """
    if not (rmin > 0):
        raise ValueError("rmin must be > 0")
    if r_ref > rmax:
        raise ValueError("r_ref must be <= rmax")
    if N_ref > size:
        raise ValueError("N_ref must be <= size")

    rng = np.random.default_rng(seed)

    # Build monotonic cumulative F(r) over [rmin, rmax]
    r_vals = np.logspace(np.log10(rmin), np.log10(rmax), ngrid).astype(float)
    F_vals = _f_proj_nfw(r_vals / rs).astype(float)

    # Ensure strictly increasing for safe inversion (tiny numerical plateaus)
    # (Usually not needed, but helps in edge cases)
    F_vals = np.maximum.accumulate(F_vals)

    F_rref = np.interp(r_ref, r_vals, F_vals)
    F_rmax = F_vals[-1]

    # Inside: NFW conditioned on r in [rmin, r_ref]  => u ~ U(F(rmin), F(r_ref))
    # Since r_vals starts at rmin, F(rmin) is F_vals[0].
    F_rmin = F_vals[0]
    u_in = rng.uniform(F_rmin, F_rref, N_ref)
    r_in = np.interp(u_in, F_vals, r_vals)

    # Outside: NFW conditioned on r in (r_ref, rmax] => u ~ U(F(r_ref), F(r_max))
    n_out = size - N_ref
    u_out = rng.uniform(F_rref, F_rmax, n_out)
    r_out = np.interp(u_out, F_vals, r_vals)

    r = np.concatenate([r_in, r_out])
    rng.shuffle(r)
    return r


def sampleMagnitudesFromFittedLF(phi_star, M_star, alpha,
                                 mag_min, mag_max,
                                 area=1.0,
                                 scatter=0.0,
                                 seed=None,
                                 bins=5000):
    """
    Compute expected number of galaxies and sample magnitudes from a Schechter function.

    Parameters:
        phi_star, M_star, alpha : float
            Best-fit Schechter parameters.
        mag_min, mag_max : float
            Magnitude range to integrate over.
        area : float
            Area (e.g., in arcmin^2) to scale expected number of galaxies.
        scatter : float
            Gaussian scatter to apply to sampled magnitudes.
        seed : int or None
            RNG seed.
        bins : int
            Number of bins for inverse transform sampling.

    Returns:
        mags : ndarray
            Array of sampled magnitudes.
    """
    rng = np.random.default_rng(seed)

    # Integrate the LF to get expected number of galaxies
    expected_density, _ = quad(schechter_mag, mag_min, mag_max,
                               args=(phi_star, M_star, alpha))
    N_expected = int(np.round(expected_density * area))

    # Prepare inverse transform sampling
    mag_grid = np.linspace(mag_min, mag_max, bins)
    pdf_vals = schechter_mag(mag_grid, phi_star, M_star, alpha)
    pdf_vals = np.clip(pdf_vals, a_min=0, a_max=None)
    cdf_vals = np.cumsum(pdf_vals)
    cdf_vals /= cdf_vals[-1]

    inv_cdf = interp1d(cdf_vals, mag_grid, bounds_error=False,
                       fill_value=(mag_min, mag_max))
    u = rng.uniform(0, 1, size=N_expected)
    mags = inv_cdf(u)

    # Add optional scatter
    if scatter > 0:
        mags += rng.normal(0, scatter, size=N_expected)

    return mags

"""
import pandas as pd
def isfloat(val):
    return all([ [any([i.isnumeric(), i in ['.','e']]) for i in val],  len(val.split('.')) == 2] )


def getClMembers(best_par='best.par',
                 fields= ('profil','x_centre','y_centre','ellipticite',
                          'angle_pos','core_radius','core_radius_kpc',
                          'cut_radius','cut_radius_kpc','v_disp','z_lens')):
    init = True
    #try:
    # opening and reading the file
    file_read = open(best_par,"r")
    lines = file_read.readlines()
    for field in fields:
        v=[]
        for line in lines:
            if field+' ' in line:
                value = [s for s in line.split() if isfloat(s) or s.isnumeric()]
                if len(value) < 2:
                    if isfloat(value[0]):
                        v.append(float(value[0]))
                    else:
                        v.append(int(value[0]))
        if init:
            df = pd.DataFrame({field:v})
            init = False
        df[field] = v
    file_read.close()
    #except:
    #    print("\nThe file doesn't exist!")
    return df
"""
