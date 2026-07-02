"""
This module contains functions to handle the photometric catalogs from Tortorelli et al. 2023, which can be found
on Vizier at https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/671/L9
"""


import pandas as pd
from pyLensLib.sersic_numba import sersic
import numpy as np

def read_photo_cat(file_path, skiprows):
    try:
        # Reading the file with semicolon as the delimiter and skipping rows as needed (all metadata)
        df = pd.read_csv(file_path, sep=';', skiprows=skiprows, comment='#')
        df = df.drop([0, 1]) # drop rows with column units and reset index
        df = df.reset_index(drop=True)
        for col in df.columns:
            df[col] = df[col].astype(float)
        #df['RAJ2000'] = df['RAJ2000'].astype(float)
        #df['DEJ2000'] = df['DEJ2000'].astype(float)
        return df
    except Exception as e:
        return str(e)

def create_sersic_from_photo_cat(df, ob, zs, band, RA_ref, DEC_ref, Npix=1000, size=100, gl=None, sizex=None, sizey=None, pcx=None, pcy=None, mask=None, save_unlensed=False,rmaxf=100):
    try:
        print ('creating the list of sersic models')
        # Extracting the relevant data from the dataframe
        ra = df['RAJ2000']
        dec = df['DEJ2000']
        # convert ra and dec to arcseconds from the reference point
        x,y = ra_dec_to_xy(ra, dec, RA_ref, DEC_ref)
        # convert band to strings for different column names
        mag = df[band].values
        reff = df['Re'+band].values
        n = df['N'+band].values
        q = df['ar'+band].values
        pa = df['PA'+band].values
        # Creating the Sersic profile
        s_list = []
        print (len(mag))
        for i in range(len(mag)):
            print (mag[i],x[i],y[i],q[i])
            flux = ob.mag2counts(mag[i])
            kwargs = {
                'n': n[i],
                'q': q[i],
                'ys1': x[i],
                'ys2': y[i],
                'pa': np.deg2rad(pa[i]),
                're': reff[i],
                'flux': flux,
                'zs': zs
            }
            s = sersic(Npix=Npix, size=size, gl=gl, save_unlensed=save_unlensed, rmaxf=rmaxf,
                       sizex=sizex, sizey=sizey, pcx=pcx, pcy=pcy, mask=mask, **kwargs)
            s_list.append(s)
        return s_list
    except Exception as e:
        return str(e)

def ra_dec_to_xy(ra, dec, ra_ref, dec_ref):
    """
    Convert RA and Dec coordinates to x, y offsets in arcseconds from a reference point.

    Parameters:
    ra (float or array-like): Right Ascension(s) of the target(s) in degrees.
    dec (float or array-like): Declination(s) of the target(s) in degrees.
    ra_ref (float): Right Ascension of the reference point in degrees.
    dec_ref (float): Declination of the reference point in degrees.

    Returns:
    x, y (tuple): Offsets in arcseconds (x and y) relative to the reference point.
    """
    # Convert all angles from degrees to radians
    dec_ref_rad = np.radians(dec_ref)

    # Compute the difference in RA and Dec
    delta_ra = ra - ra_ref
    delta_dec = dec - dec_ref

    # Convert RA difference to arcseconds, accounting for cos(dec) to get true angular separation
    x = - delta_ra * np.cos(dec_ref_rad) * 3600  # RA difference scaled by cos(Dec_ref) in arcseconds
    y = delta_dec * 3600  # Dec difference in arcseconds

    return x, y

if __name__ == '__main__':
    from pyLensLib.observation import observation
    import matplotlib.pyplot as plt
    # import LogNorm
    from matplotlib.colors import LogNorm
    # Usage example

    fov = 200.0
    file_path = '../Test/M0416.tsv'
    skip_rows = 40  # Adjust this based on the structure of your file
    df = read_photo_cat(file_path, skip_rows)
    ob = observation(size=fov, Npix=1000, zp=23.9, mlim=24.5, rap=1.3, sn=10.0)
    se=create_sersic_from_photo_cat(df,ob,1.0,'F814W',64.038142, -24.067472, Npix=1000,
                                    size=fov, gl=None, sizex=None, sizey=None, pcx=None, pcy=None,
                                    mask=None, save_unlensed=False,rmaxf=100)

    #se = create_sersic_from_photo_cat(df, ob, 1.0, 'F814W', 342.183210, -44.530878, Npix=1000,
    #                                  size=fov, gl=None, sizex=None, sizey=None, pcx=None, pcy=None,
    #                                  mask=None, save_unlensed=False, rmaxf=100)
    print (type(se), len(se))
    image = np.zeros_like(se[0].image)
    for s in se:
        image += s.image

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image, origin='lower', extent=[-50, 50, -50, 50],norm=LogNorm())
    plt.show()


