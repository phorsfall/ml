import cPickle as pickle
import os.path
import collections
import functools
import itertools
import operator

import numpy as np

from ml import utils, data, optimize, mnist, rbm
from ml.sigmoid import logistic

OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')
DATA_PATH = os.path.expanduser('~/Development/ml/datasets')

# The first arg passed to namedtuple needs to match the var name
# otherwise pickling fails.
params_tuple = collections.namedtuple('params_tuple', 'W_r1 b_r1 W_r2 b_r2 W_g1 b_g1 W_g2 b_g2 W v_bias h_bias')

def load_params(path, filename):
    with(open(os.path.join(path, filename), 'rb')) as f:
        return params_tuple._make(pickle.load(f))

def sample_v_softmax(rbm, h, h_mean, k):
    """
    Sample the visible units, treating the k right-most units as a
    softmax group.
    """
    # Top down activity for all units.
    a = h.dot(rbm.W) + rbm.v_bias
    # Softmax units.
    u = a[:,-k:] # Un-normalized log probabilities.
    log_prob = u - np.log(np.sum(np.exp(u), 1))[:,np.newaxis]
    prob = np.exp(log_prob)
    labels = sample_softmax(prob)
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

sample_v = functools.partial(sample_v_softmax, k=mnist.NUM_CLASSES)    

def up_down(params, data, weight_penalty=0, rng=np.random):
    """
    The up-down algorithms as described in "A fast learning algorithm
    for deep belief nets".
    """
    inputs, targets = data.inputs, data.targets
    num_cases = inputs.shape[0]

    # Wake/positive phase.
    wake_hid1_probs = logistic(inputs.dot(params.W_r1.T) + params.b_r1)
    wake_hid1_states = wake_hid1_probs > rng.random(wake_hid1_probs.shape)

    wake_hid2_probs = logistic(wake_hid1_states.dot(params.W_r2.T) + params.b_r2)
    wake_hid2_states = wake_hid2_probs > rng.random(wake_hid2_probs.shape)

    # Ideally I'd like to use rbm.cd to do the training of the
    # top-level rbm, but as it stands although I could use it to
    # compute the gradient I don't get back the states of the hidden
    # units which I need to perform the down pass. Perhaps this could
    # be solved if the gibbs chain could be passing in to rbm.cd as an
    # parameter.

    rbm_params = rbm.params(params.W, params.v_bias, params.h_bias)
    
    k = 8 # cd_k
    gc = rbm.gibbs_chain(np.hstack((wake_hid2_states, targets)), rbm_params, rbm.sample_h, sample_v, k+1)

    v0, h0, h0_mean = gc.next()
    v1, h1, h1_mean = itertools.islice(gc, k-1, None).next()

    pos_grad = rbm.neg_free_energy_grad(params, (v0, h0, h0_mean))
    neg_grad = rbm.neg_free_energy_grad(params, (v1, h1, h1_mean))

    rbm_grad = map(operator.sub, neg_grad, pos_grad)

    # sleep_hid2_states really is states as the sample_v_softmax
    # function in this file does do sampling.
    sleep_hid2_states = v1[:,:-mnist.NUM_CLASSES]
    sleep_hid1_probs = logistic(sleep_hid2_states.dot(params.W_g2) + params.b_g2)
    sleep_hid1_states = sleep_hid1_probs > rng.random(sleep_hid1_probs.shape)
    sleep_vis_probs = logistic(sleep_hid1_states.dot(params.W_g1) + params.b_g1)

    # Predictions.
    p_sleep_hid2_states = logistic(sleep_hid1_states.dot(params.W_r2.T) + params.b_r2)
    p_sleep_hid1_states = logistic(sleep_vis_probs.dot(params.W_r1.T) + params.b_r1)
    p_vis_probs = logistic(wake_hid1_states.dot(params.W_g1) + params.b_g1)
    p_hid1_probs = logistic(wake_hid2_states.dot(params.W_g2) + params.b_g2)
    
    # Updates to generative parameters.
    W_g1_grad = wake_hid1_states.T.dot(inputs - p_vis_probs) / num_cases
    b_g1_grad = np.mean(inputs - p_vis_probs, 0)
    
    W_g2_grad = wake_hid2_states.T.dot(wake_hid1_states - p_hid1_probs) / num_cases
    b_g2_grad = np.mean(wake_hid1_states - p_hid1_probs, 0)

    # Updates to recognition parameters.
    W_r2_grad = sleep_hid1_states.T.dot(sleep_hid2_states - p_sleep_hid2_states).T / num_cases
    b_r2_grad = np.mean(sleep_hid2_states - p_sleep_hid2_states, 0)
    W_r1_grad = sleep_vis_probs.T.dot(sleep_hid1_states - p_sleep_hid1_states).T / num_cases
    b_r1_grad = np.mean(sleep_hid1_states - p_sleep_hid1_states, 0)

    # Reconstruction error.
    cost = np.sum((inputs - sleep_vis_probs) ** 2) / num_cases
    
    grad = (-W_r1_grad, -b_r1_grad,
            -W_r2_grad, -b_r2_grad,
            -W_g1_grad, -b_g1_grad,
            -W_g2_grad, -b_g2_grad,
            rbm_grad[0], rbm_grad[1], rbm_grad[2])

    if weight_penalty > 0:
        grad = tuple(g_i + weight_penalty * p_i
                     for p_i, g_i
                     in zip(params, grad))

    return cost, grad

# Each run listed below started from the parameters from the line
# above and using the meta-parameteres listed.

# mnist_dbm_pretrain.zip.
# 1357409197 - cd_3, lr 0.01, momentum (0.5, 0.9, 10), 50 epochs
# 1357428588 - cd_3, lr 0.01, momentum (0.5, 0.9, 10), 50 epochs
# 1357463280 - cd_5, lr 0.001, momentum (0.5, 0.9, 10), 50 epochs
# 1357497070 - cd_10, lr 0.00001, momentum (0.5, 0.9, 10), 25 epochs
# 1357570303 - cd_8, lr 0.001, m (0.5, 0.9, 10), 50 ep, wp 0.0002

initial_params = load_params(os.path.join(OUTPUT_PATH, '1357497070'), '24.pickle')

training, validation = mnist.load_training_dataset(DATA_PATH)
batches = data.BatchIterator(training)

epochs = 50
learning_rate = 0.001
momentum = optimize.linear(0.5, 0.9, 10)
weight_penalty = 0.0002

save_params = utils.save_params_hook(OUTPUT_PATH)

def f(params, data):
    return up_down(params, data, weight_penalty=weight_penalty)

opt_params = optimize.sgd(f, initial_params, batches, epochs, learning_rate,
                          momentum=momentum,
                          post_epoch=save_params)
