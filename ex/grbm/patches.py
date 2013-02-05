import os.path
import cPickle as pickle
import functools

import numpy as np

from ml import mnist, utils, optimize, rbm, grbm, data, meta
from ml import regularization as reg

DATA_PATH = os.path.expanduser("~/Development/ml/datasets")

def load_inputs():
    # This data set is sampled from Bruno Olshausen's unwhitened
    # natural images: http://redwood.berkeley.edu/bruno/sparsenet/
    # Send me an email (see GitHub profile page) if you'd like a copy
    # of the exact patches I used here.
    fn = 'natural-image-patches-20k-8x8.pickle'
    with(open(os.path.join(DATA_PATH, fn))) as f:
        data = pickle.load(f)
    # print data['desc']
    return data['inputs']

def ex1(inputs):
    """
    Gaussian/Bernoulli RBM.
    """
    # Can learns edge detector like filters although learning is quite
    # slow and learning is very sensitive to meta-parameter selection.

    # Momentum seems neccesary, without it it's difficult to learn
    # anything.

    # When learning on whitened data setting the fudge factor to
    # something around 0.1 was important. Setting it much too much
    # lower causes point filters to be learned.

    # Learning does happen if you don't use whitening, but the
    # features tend to be less localized when compared to the learned
    # with whitening. Interestingly the reconstruction error is lower
    # without whitening, but I suspect I'm comparing apples with
    # oranges there.

    # With only 25 hidden units I couldn't find a way to learn
    # anything much. Contrast this to an autoencoder which does seem
    # to learn filters in a similar situation.

    # error (100 epochs) = 25.492607
    # error (500 epochs) = 24.096789

    # See ex1.png.

    inputs = utils.remove_dc(inputs)
    inputs, zca = utils.zca_white(inputs, 0.1)
    batches = data.BatchIterator(inputs, 50)
    num_vis = 64
    num_hid = 100
    epochs = 500
    initial_params = grbm.initial_params(num_hid, num_vis, 0.05)

    sample_v = functools.partial(grbm.sample_v, add_noise=False)
    neg_free_energy_grad = functools.partial(grbm.neg_free_energy_grad,
                                             learn_sigma=False)

    def f(params, inputs):
        return rbm.cd(params, inputs,
                      grbm.sample_h, sample_v,
                      neg_free_energy_grad)
  
    learning_rate = 0.01
    momentum = meta.step(0.5, 0.9, 5)
    return optimize.sgd(f, initial_params, batches,
                        epochs, learning_rate, momentum)


def ex2(inputs):
    """
    Gaussian/NReLU RBM.
    """
    # Using noisy rectified linear units for the visibles speeds up
    # learning dramatically. The reconstruction error after a single
    # epoch is lower (21.6986) than after 500 epochs in ex1.

    # The filters learned have less noisy backgrounds than those
    # learned in ex1.

    # error (100 epochs) = 15.941531
    # error (500 epochs) = 15.908922

    # See ex2.png.

    inputs = utils.remove_dc(inputs)
    inputs, zca = utils.zca_white(inputs, 0.1)
    batches = data.BatchIterator(inputs, 50)
    num_vis = 64
    num_hid = 100
    epochs = 500
    initial_params = grbm.initial_params(num_hid, num_vis, 0.05)

    sample_v = functools.partial(grbm.sample_v, add_noise=False)
    neg_free_energy_grad = functools.partial(grbm.neg_free_energy_grad,
                                             learn_sigma=False)

    def f(params, inputs):
        return rbm.cd(params, inputs,
                      grbm.sample_h_noisy_relu, sample_v,
                      neg_free_energy_grad)

    learning_rate = 0.01
    momentum = meta.step(0.5, 0.9, 5)
    return optimize.sgd(f, initial_params, batches,
                        epochs, learning_rate, momentum)


def ex3(inputs):
    """
    Gaussian/NReLU RBM with learned visible variances.
    """
    # I found it essential to add noise/sample from the visible units
    # during reconstruction. If I don't do this the variances increase
    # at each epoch (I'd expect them to decrease during learning from
    # their initial value of one) as does the error.
    
    # This result was obtained by running SGD without using momentum.
    # The default momentum schedule set-up a big oscillation which
    # caused the error to increase over a few epochs after which we
    # learning appeared to be stuck out on a plateau. More modest
    # schedules (such as 0.1 for the first 10 epochs, 0.2 thereafter)
    # allow learning but they don't result in any improvement in
    # error.
    
    # The variances learned are all very similar. Their mean is 0.39,
    # their standard deviation is 0.04.

    # A quick test suggests that smaller initial weights lead to a
    # slightly lower reconstruction error.

    # error (100 epochs) = 7.401834
    # error (500 epochs) = 7.245722

    # See ex3.png.

    inputs = utils.remove_dc(inputs)
    inputs, zca = utils.zca_white(inputs, 0.1)
    batches = data.BatchIterator(inputs, 50)
    num_vis = 64
    num_hid = 100
    epochs = 500
    initial_params = grbm.initial_params(num_hid, num_vis, 0.05)

    def f(params, inputs):
        return rbm.cd(params, inputs,
                      grbm.sample_h_noisy_relu, grbm.sample_v,
                      grbm.neg_free_energy_grad)
    
    learning_rate = 0.01
    momentum = 0
    return optimize.sgd(f, initial_params, batches,
                        epochs, learning_rate, momentum)


np.random.seed(32)
inputs = load_inputs()[0:10000]
params = ex3(inputs)
