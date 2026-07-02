# create catalog of poststamps galaxies

from pyLensLib.srccatalog import srccatalog

kwargs = {
    'fov': 100.0,
    'filtern': 'BPZ/HST_ACS_WFC_F606W.res',
    'useband': 'f775w',
    'recal': 'yes',
    'maglim': 28.0,
    'udfdir': '/Users/maxmen3/stiva/HUDF/',
    'homedir': '/Users/maxmen3/CODES/bpz-1.99.3/SED/'
}


#cat = srccatalog(filename='test_xdf.cat',**kwargs)
cat = srccatalog(**kwargs)
df = cat.get_dataframe()
print(df.head())