import os.path
import functools
import itertools
import collections
import cPickle as pickle

import numpy as np

from ml import mnist, utils, optimize, rbm, data
from ml.sigmoid import logistic

print '***eval***'

# This is my first attempt at generating fantasies from a deep belief
# net. The code is pretty messy so I might just dump it, although I
# learnt a bunch from it.

# Generating from this model (that is, without fine-tuning) isn't all
# that great.

DATA_PATH = os.path.expanduser('~/Development/ml/datasets')
OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

params_tuple = collections.namedtuple('params_tuple', 'W_r1 b_r1 W_r2 b_r2 W_g1 b_g1 W_g2 b_g2 W v_bias h_bias')

def load_params(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# TODO: Maybe create a rbm_softmax module for this if it works out.
# TODO: Need to figure out how determine whether this and other
# sample_v functions should perform sampling. At the moment,
# rbm.sample_v never does, but this only works for cd1, it won't work
# when generating samples etc.
def sample_v_softmax(rbm, h, h_mean, k, clamped_labels=None):
    """
    Sample the visible units, treating the k right-most units as a
    softmax group.
    """
    # Top down activity for all units.
    a = h.dot(rbm.W) + rbm.v_bias
    # Softmax units.
    if clamped_labels is None:
        u = a[:,-k:] # Un-normalized log probabilities.
        log_prob = u - np.log(np.sum(np.exp(u), 1))[:,np.newaxis]
        prob = np.exp(log_prob)
        labels = sample_softmax(prob)
    else:
        labels = clamped_labels
    # Logistic units.
    v_mean = logistic(a[:,:-k])
    v = v_mean > np.random.random(v_mean.shape)
    return np.hstack((v, labels))

def sample_softmax(prob):
    num_cases = prob.shape[0]
    out = np.zeros(prob.shape)
    # Can this be vectorized?
    for i in xrange(num_cases):
        out[i] = np.random.multinomial(1, prob[i], 1)
    return out

def downward_pass(visible_activations, params):
    """
    Perform a deterministic downward pass starting as the visible
    units of the top-level associative memory.
    """
    # TODO: This could probably reuse the various sample_v functions.
    k = mnist.NUM_CLASSES
    # Downward pass.
    # Not sure whether to sample here. So far, it doesn't seem to make
    # much difference.
    v2_mean = visible_activations[:,0:-k]
    #v2 = v2_mean > np.random.random(v2_mean.shape)
    v2 = v2_mean
    v1_mean = logistic(v2.dot(params.W_g2) + params.b_g2)
    #v1 = v1_mean > np.random.random(v1_mean.shape)
    v1 = v1_mean
    pixels = logistic(v1.dot(params.W_g1) + params.b_g1)
    return pixels


np.random.seed(32)

# TODO: Maybe pickle these as a single file. Is a tuple ok?
# Load the parameters from greedy pre-training of the RBMs.
params = load_params(os.path.join(OUTPUT_PATH, '1357570303', '49.pickle'))

# TODO: Need to figure out the naming of layers etc.

num_hid2 = 500

n = 10 # Number of fantasies to generate.

clamp_digit = 2
clamped_labels = np.repeat(np.eye(mnist.NUM_CLASSES)[[clamp_digit]], n, 0)
clamped_labels = np.eye(10)

uniform_initial_labels = np.ones((n, mnist.NUM_CLASSES)) / mnist.NUM_CLASSES

# Initializing the chain by doing a bottom up pass from random pixels
# seems to work better than randomly initializing the visible units of
# the top-level RBM.
# initial_pen = np.random.random((n, num_hid2))
rand_pixels = np.random.random((n, 784))
rand_hid1 = logistic(rand_pixels.dot(params.W_r1.T) + params.b_r1)
rand_hid1 = rand_hid1 > np.random.random(rand_hid1.shape)
rand_hid2 = logistic(rand_hid1.dot(params.W_r2.T) + params.b_r2)
rand_hid1 = rand_hid2 > np.random.random(rand_hid2.shape)
initial_pen = rand_hid2


#initial_v = np.hstack((initial_pen, uniform_initial_labels))
initial_v = np.hstack((initial_pen, clamped_labels))

sample_v = functools.partial(sample_v_softmax,
                             k=mnist.NUM_CLASSES,
                             clamped_labels=clamped_labels)

gc = rbm.gibbs_chain(initial_v,
                     rbm.params(params.W, params.v_bias, params.h_bias),
                     rbm.sample_h,
                     sample_v)

# TODO: This is similar to rbm.fantasize. Refactor.
# Start, end, step
gen = itertools.islice(gc, 100, None, 8)
# Count.
gen = itertools.islice(gen, 250)
# We only want the state of the visibles.
gen = itertools.imap(lambda sample: sample[0], gen)
# Downward pass.
gen = itertools.imap(lambda v: downward_pass(v, params), gen)
# Tile.
gen = itertools.imap(functools.partial(utils.tile, grid_shape=(2, 5)), gen)
