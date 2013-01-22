# Various functions which I find useful when working interactively
# with matplotlib.

# The intention is that this module can be imported into the top-level
# namespace as pylab is. That is:
# from ml.pylab import *

import matplotlib
from ml import utils

def gs():
    """
    Default imshow to using a grayscale color map and disable
    interpolation. This is useful when viewing weights, MNIST digits
    etc.
    """
    matplotlib.rcParams['image.cmap'] = 'gray'
    matplotlib.rcParams['image.interpolation'] = 'none'

def imtile(X, *args, **kwargs):
    if 'v' in kwargs:
        v = kwargs['v']
        kwargs['vmin'] = -v
        kwargs['vmax'] = v
        del kwargs['v']
    matplotlib.pyplot.imshow(utils.tile(X), *args, **kwargs)
