import os.path
import functools
import itertools
import collections
import operator
import tempfile

import numpy as np

from ml import mnist, utils, optimize, rbm, dbn, data, parameters
from ml.sigmoid import logistic

np.random.seed(32)

DATA_PATH = os.path.expanduser('~/Development/ml/datasets')
OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

sample_v_softmax = functools.partial(rbm.sample_v_softmax, k=mnist.NUM_CLASSES)

def up_pass(params, pixels):
    """
    Perform and upward pass from the visible pixels to the visible
    units of the top-level RBM.
    """
    # This is deterministic. (i.e. It uses the real-valued
    # probabilities rather than sampling.)
    hid1_mean = logistic(pixels.dot(params[0].W_r) + params[0].b_r)
    hid2_mean = logistic(hid1_mean.dot(params[1].W_r) + params[1].b_r)
    return hid2_mean

def down_pass(params, v):
    """
    Perform a deterministic downward pass from the visible units of
    the top-level RBM to the visible pixels.
    """
    # The visible units of the top-level RBM include a softmax group
    # which is not directly connected to the visible pixels.
    hid2_mean = v[:,mnist.NUM_CLASSES:]
    hid1_mean = logistic(hid2_mean.dot(params[1].W_g) + params[1].b_g)
    vis_mean = logistic(hid1_mean.dot(params[0].W_g) + params[0].b_g)
    return vis_mean

# inputs = mnist.load_inputs(DATA_PATH, mnist.TRAIN_INPUTS)
# labels = mnist.load_labels(DATA_PATH, mnist.TRAIN_LABELS)
# inputs, _, labels = data.balance_classes(inputs, labels, mnist.NUM_CLASSES)

def generate(params):
    dbn_params = dbn.stack_params(params)

    # 10 fantasies.
    # initial_pixels = np.zeros((10, 28**2))
    # initial_pixels = inputs[0:10]

    # Clamp the softmax units, one for each class.
    sample_v_softmax_clamped = functools.partial(sample_v_softmax,
                                                 labels=np.eye(10))

    # Perform an upward pass from the pixels to the visible units of
    # the top-level RBM.
    # initial_v = np.hstack((
    #         np.eye(10),
    #         up_pass(dbn_params, initial_pixels)))

    initial_v = np.hstack((
            np.eye(10),
            np.random.random((10, dbn_params[-1].W.shape[1] - 10))))

    # Initialize the gibbs chain.
    gc = rbm.gibbs_chain(initial_v,
                         dbn_params[-1],
                         rbm.sample_h,
                         sample_v_softmax_clamped)

    tile_2_by_5 = functools.partial(utils.tile, grid_shape=(2, 5))

    gen = itertools.islice(gc, 1, None, 1)
    gen = itertools.islice(gen, 2000)
    gen = itertools.imap(operator.itemgetter(1), gen)
    gen = itertools.imap(lambda v: down_pass(dbn_params, v), gen)
    gen = itertools.imap(tile_2_by_5, gen)

    # Save to disk.
    utils.save_images(gen, tempfile.mkdtemp(dir=OUTPUT_PATH))


params = parameters.load(OUTPUT_PATH, timestamp=1358586160)
generate(params)
