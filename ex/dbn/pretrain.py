import os.path
import functools

import numpy as np

from ml import mnist, utils, optimize, data, rbm
from ml.sigmoid import logistic

DATA_PATH = os.path.expanduser('~/Development/ml/datasets')
OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

np.random.seed(32)

# Maybe create a rbm_softmax module for this if it works out.
def sample_v_softmax(rbm, h, h_mean, k):
    """
    Sample the visible units, treating the k right-most units as a
    softmax group.
    """
    # TODO: Perform sampling.
    # Top down activity for all units.
    a = h.dot(rbm.W) + rbm.v_bias
    # Softmax units.
    u = a[:,-k:] # Un-normalized log probabilities.
    log_prob = u - np.log(np.sum(np.exp(u), 1))[:,np.newaxis]
    prob = np.exp(log_prob)
    # Logistic units.
    v = logistic(a[:,:-k])
    return np.hstack((v, prob))

sample_v = functools.partial(sample_v_softmax, k=mnist.NUM_CLASSES)    

def sgd(f, initial_params, batches):
    save_params = utils.save_params_hook(OUTPUT_PATH)
    return optimize.sgd(f, initial_params, batches,
                        epochs, learning_rate, momentum,
                        post_epoch=save_params)

# Optimization objective for the first two layers.
def rbm_obj(params, inputs):
    return rbm.cd(params, inputs,
                  rbm.sample_h,
                  rbm.sample_v,
                  rbm.neg_free_energy_grad,
                  weight_penalty)

# Optimization objective for the top layer.
def rbm_softmax_obj(params, inputs):
    return rbm.cd(params, inputs,
                  rbm.sample_h,
                  sample_v,
                  rbm.neg_free_energy_grad,
                  weight_penalty)

epochs = 50
momentum = optimize.linear(0.5, 0.9, 10)
learning_rate = 0.1
weight_penalty = 0.0002

inputs = mnist.load_inputs(DATA_PATH, mnist.TRAIN_INPUTS)
labels = mnist.load_labels(DATA_PATH, mnist.TRAIN_LABELS)
targets = data.targets_from_labels(labels, mnist.NUM_CLASSES)

num_pixels = inputs.shape[1]
num_hid1 = 500
num_hid2 = 500
num_top = 2000

batches = data.BatchIterator(inputs)
initial_params = rbm.initial_params(num_hid1, num_pixels)
params = sgd(rbm_obj, initial_params, batches)

inputs = logistic(inputs.dot(params.W.T) + params.h_bias)
batches = data.BatchIterator(inputs)
initial_params = rbm.initial_params(num_hid2, num_hid1)
params = sgd(rbm_obj, initial_params, batches)

inputs = logistic(inputs.dot(params.W.T) + params.h_bias)
batches = data.BatchIterator(np.hstack((inputs, targets)))
initial_params = rbm.initial_params(num_top, num_hid2 + mnist.NUM_CLASSES)
params = sgd(rbm_softmax_obj, initial_params, batches)
