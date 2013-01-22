import collections
import functools
import itertools
import operator
import os.path

import numpy as np

from ml import utils, data, optimize, mnist, rbm, dbn, parameters, meta
from ml import regularization as reg
from ml.sigmoid import logistic

OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')
DATA_PATH = os.path.expanduser('~/Development/ml/datasets')

def contrastive_wake_sleep(params, data, weight_decay=None, cd_k=1):
    inputs, targets = data.inputs, data.targets
    num_cases = inputs.shape[0]

    # Turn the single tuple of parameters into something easier to
    # work with.
    dbn_params = dbn.stack_params(params)
    grad = []

    # Wake phase.
    wake_hid1_states = rbm.sample_bernoulli(logistic(inputs.dot(dbn_params[0].W_r) + dbn_params[0].b_r))
    wake_hid2_states = rbm.sample_bernoulli(logistic(wake_hid1_states.dot(dbn_params[1].W_r) + dbn_params[1].b_r))

    # Contrastive divergence.
    gc = rbm.gibbs_chain(np.hstack((targets, wake_hid2_states)),
                         dbn_params[-1],
                         rbm.sample_h,
                         sample_v_softmax,
                         cd_k + 1)

    pos_sample = gc.next()
    if cd_k == 1:
        neg_sample = gc.next()
    else:
        recon_sample = gc.next()
        neg_sample = itertools.islice(gc, cd_k - 2, None).next()

    # Sleep phase.
    sleep_hid2_states = neg_sample[0][:,mnist.NUM_CLASSES:]
    sleep_hid1_states = rbm.sample_bernoulli(logistic(sleep_hid2_states.dot(dbn_params[1].W_g) + dbn_params[1].b_g))
    sleep_vis_probs = logistic(sleep_hid1_states.dot(dbn_params[0].W_g) + dbn_params[0].b_g)

    # Predictions.
    p_sleep_hid2 = logistic(sleep_hid1_states.dot(dbn_params[1].W_r) + dbn_params[1].b_r)
    p_sleep_hid1 = logistic(sleep_vis_probs.dot(dbn_params[0].W_r) + dbn_params[0].b_r)
    p_wake_vis = logistic(wake_hid1_states.dot(dbn_params[0].W_g) + dbn_params[0].b_g)
    p_wake_hid1 = logistic(wake_hid2_states.dot(dbn_params[1].W_g) + dbn_params[1].b_g)

    # Gradients.
    # Layer 0.
    W_r_grad = sleep_vis_probs.T.dot(p_sleep_hid1 - sleep_hid1_states) / num_cases
    b_r_grad = np.mean(p_sleep_hid1 - sleep_hid1_states, 0)
    W_g_grad = wake_hid1_states.T.dot(p_wake_vis - inputs) / num_cases
    b_g_grad = np.mean(p_wake_vis - inputs, 0)
    grad.extend([W_r_grad, b_r_grad, W_g_grad, b_g_grad])

    # Layer 1.
    W_r_grad = sleep_hid1_states.T.dot(p_sleep_hid2 - sleep_hid2_states) / num_cases
    b_r_grad = np.mean(p_sleep_hid2 - sleep_hid2_states, 0)
    W_g_grad = wake_hid2_states.T.dot(p_wake_hid1 - wake_hid1_states) / num_cases
    b_g_grad = np.mean(p_wake_hid1 - wake_hid1_states, 0)
    grad.extend([W_r_grad, b_r_grad, W_g_grad, b_g_grad])
    
    # Top-level RBM.
    pos_grad = rbm.neg_free_energy_grad(dbn_params[-1], pos_sample)
    neg_grad = rbm.neg_free_energy_grad(dbn_params[-1], neg_sample)
    rbm_grad = map(operator.sub, neg_grad, pos_grad)
    grad.extend(rbm_grad)

    # Weight decay.
    if weight_decay:
        weight_grad = (weight_decay(p)[1] for p in params)
        grad = map(operator.add, grad, weight_grad)

    # One-step reconstruction error.
    if cd_k == 1:
        recon = sleep_vis_probs
    else:
        # Perform a determisitic down pass from the first sample of
        # the Gibbs chain in order to compute the one-step
        # reconstruction error.
        recon_hid2_probs = recon_sample[1][:,mnist.NUM_CLASSES:]
        recon_hid1_probs = rbm.sample_bernoulli(logistic(recon_hid2_probs.dot(dbn_params[1].W_g) + dbn_params[1].b_g))
        recon = logistic(recon_hid1_probs.dot(dbn_params[0].W_g) + dbn_params[0].b_g)

    error = np.sum((inputs - recon) ** 2) / num_cases

    return error, grad

np.random.seed(1234)

# data
inputs = mnist.load_inputs(DATA_PATH, mnist.TRAIN_INPUTS)
labels = mnist.load_labels(DATA_PATH, mnist.TRAIN_LABELS)
inputs, targets, labels = data.balance_classes(inputs, labels, mnist.NUM_CLASSES)
n = 54200
inputs = inputs[0:n]
targets = targets[0:n]
labels = labels[0:n]
dataset = data.dataset(inputs, targets, labels)
batches = data.BatchIterator(dataset)

# meta
epochs = 50
learning_rate = 0.1
cd_k = 15
weight_decay = reg.l2(0.0002)
momentum = 0 #0.1

# Load and stack params from initial pre-training.
# initial_params = dbn.params_from_rbms([parameters.load(OUTPUT_PATH, t)
#                                        for t in (1358445776, 1358451846, 1358456203)])

# Params after first 50 epochs of fine tuning. (lr 0.1, p 0.0, cd_k=10)
initial_params = parameters.load(OUTPUT_PATH, timestamp=1358541318)

# The sampling function used by the top-level RBM.
sample_v_softmax = functools.partial(rbm.sample_v_softmax, k=mnist.NUM_CLASSES)

# Optimization objective.
def f(params, data):
    return contrastive_wake_sleep(params, data, weight_decay, cd_k)

output_dir = utils.make_output_directory(OUTPUT_PATH)
save_params = parameters.save_hook(output_dir)

params = optimize.sgd(f, initial_params, batches, epochs, learning_rate, momentum,
                      post_epoch=save_params)
