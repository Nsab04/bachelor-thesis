from pyLensLib.srccatalog import srccatalog
import matplotlib

matplotlib.use('TkAgg')

kwargs = {
    'fov': 200.0,
    'filtern': 'BPZ/HST_ACS_WFC_F606W.res',
    'useband': 'f775w',
    'recal': 'yes',
    'maglim': 28.0,
    'udfdir': '/Users/massimo/stiva/HUDF/',
    'homedir': '/Users/massimo/CODES/bpz-1.99.3/SED/'
}


cat = srccatalog(filename='test.cat',**kwargs)