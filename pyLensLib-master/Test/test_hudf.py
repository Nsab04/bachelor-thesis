# this code reads the postage stamps of the HUDF and pack them into a hdf5 file.
# In the process, postage stamps are made homogeneous (224x224 pixels each)
# The original stamp size is stored in hudf_size

import glob
from astropy.io import fits as fits
from tqdm import tqdm
from skimage.transform import resize
import numpy as np

# import the dataset of postage stamps
path='/Users/maxmen3/stiva/HUDF/archive_max30/'
udfdir='/Users/maxmen3/stiva/HUDF/'
id,coe_id,f225w,f275w,f336w,f435w,f606w,f775w,f850lp,f105w,f125w,f140w,f160w,fwhm,areaf,z,st=np.loadtxt(udfdir+'uvudf_rafelski_2015.dat',unpack=True,usecols=[0,1,6,7,8,9,10,11,12,13,14,15,16,73,74,80,85])


gal_ima=[]
gal_ima_res=[]
gal_id=[]
gal_size=[]
gal_z=[]
gal_bpz_templ=[]
gal_mags =[]
fits_file=glob.glob(path+'*.fits')

ii=0
for f in tqdm(fits_file):
    funder=f.find('obj_')
    fpoint=f.find('.fits')
    gal_id.append(f[funder+4:fpoint])
    isel= id == np.float64(f[funder+4:fpoint])
    gal_z.append(z[isel][0])

    hdul=fits.open(f,memmap=False)#,1,header=True)
    gal_size.append(hdul[0].data.shape[0]*0.03)
    if ii == 0:
        print (f,f[funder+4:fpoint],hdul[0].data.shape[0]*0.03,z[isel])
        ii+=1
    b_image=resize(hdul[0].data, (224,224), mode='reflect')
    v_image=resize(hdul[1].data, (224,224), mode='reflect')
    r_image=resize(hdul[2].data, (224,224), mode='reflect')
    i_image=resize(hdul[3].data, (224,224), mode='reflect')
    gal_ima.append([hdul[0].data,hdul[1].data,hdul[2].data,hdul[3].data])
    gal_ima_res.append([b_image,v_image,i_image,r_image])
    gal_bpz_templ.append(st[isel][0])
    gal_mags.append([f225w[isel][0],f275w[isel][0],f336w[isel][0],f435w[isel][0],f606w[isel][0],
                    f775w[isel][0],f850lp[isel][0],f105w[isel][0],f125w[isel][0],f140w[isel][0],f160w[isel][0]])

    hdul.close()

# storing the following information:
# hudf_resized: postage stamp
# hudf_ids: source id in the HUDF catalog (Rafelski et al. 2015)
# hudf_size: side length of the postage stamp

print (np.array(gal_id).astype(int).shape)
exit()

import h5py
hf = h5py.File('hudf_dataset.h5', 'w')
hf.create_dataset("hudf_resized",data=gal_ima_res,dtype = 'float32')
hf.create_dataset("hudf_ids",data=np.array(gal_id).astype(int))#'|S7')
hf.create_dataset("hudf_size",data=gal_size,dtype = 'float32')
hf.create_dataset("hudf_z",data=gal_z,dtype = 'float32')
hf.create_dataset("hudf_template",data=gal_bpz_templ,dtype = 'float32')
hf.create_dataset("hudf_mags",data=gal_mags,dtype = 'float32')
hf.close()
