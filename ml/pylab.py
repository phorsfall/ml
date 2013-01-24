# Various functions which I find useful when working interactively
# with matplotlib.

# The intention is that this module can be imported into the top-level
# namespace as pylab is. That is:
# from ml.pylab import *

import matplotlib
from ml import utils

def gs():
    """
    Default imshow to using a grayscale color map. This is useful when
    viewing weights, MNIST digits etc.
    """
    matplotlib.rcParams['image.cmap'] = 'gray'

def imtile(X, *args, **kwargs):
    if 'v' in kwargs:
        v = kwargs['v']
        kwargs['vmin'] = -v
        kwargs['vmax'] = v
        del kwargs['v']
    tile_kwargs = {}
    if 'channel_count' in kwargs:
        tile_kwargs['channel_count'] = kwargs['channel_count']
        del kwargs['channel_count']
    matplotlib.pyplot.imshow(utils.tile(X, **tile_kwargs), *args, **kwargs)

def cimtile(X, *args, **kwargs):
    kwargs['channel_count'] = 3
    imtile(X, *args, **kwargs)
