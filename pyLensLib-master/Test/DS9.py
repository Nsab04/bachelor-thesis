"""
Finds the counter-images of an extended source, defined starting from a region in DS9 (XPA messaging system needed, https://github.com/ericmandel/xpa)
To do in DS9: WCS -> Degrees
"""

import argparse
import pyregion
import pyds9

ds9 = pyds9.DS9()

#FITS PROPRIETIES

hdu_fits = ds9.get_pyfits()
header = hdu_fits[0].header
image = hdu_fits[0].data

crpix1 = header['CRPIX1']
crpix2 = header['CRPIX2']
crval1 = header['CRVAL1']
crval2 = header['CRVAL2']
cd1_1 = header['CD1_1']
cd1_2 = header['CD1_2']
cd2_1 = header['CD2_1']
cd2_2 = header['CD2_2']

# PARAMETERS OF THE ELLIPSE (This is for a single region, generalization to multiple region is very simple)

z = 1.0

regions = pyregion.parse(ds9.get('regions'))

ra_reg = regions[0].coord_list[0]
dec_reg = regions[0].coord_list[1]
a_reg = regions[0].coord_list[2] * 3600
b_reg = regions[0].coord_list[3] * 3600
theta_reg = regions[0].coord_list[4]
name = regions[0].attr[1]['text']
color = regions[0].attr[1]['color']
