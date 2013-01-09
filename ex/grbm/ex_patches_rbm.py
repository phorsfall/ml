import os.path
import functools
import cPickle as pickle

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'none'

from pylab import *
ion()

import mnist
import utils
import optimize
import rbm
import grbm

reload(mnist)
reload(utils)
reload(optimize)
reload(rbm)
reload(grbm)

DATA_PATH = os.path.expanduser("~/Datasets")

def load_inputs():
    fn = 'natural-image-patches-20k-8x8.pickle'
    with(open(os.path.join(DATA_PATH, fn))) as f:
        data = pickle.load(f)
        return data['inputs']

def train_rbm(batches, params, epochs,
              use_noisy_relu, learn_sigma,
              add_reconstruction_noise,
              use_momentum):
    # This is just a convenient wrapper around the contrastive
    # divergence optimization objective and stochastic gradient
    # descent to make it easier to try different configurations.

    # Turn the config parameters into the correct functions to be
    # passed to cd1.
    if use_noisy_relu:
        sample_h = grbm.sample_h_noisy_relu
    else:
        sample_h = grbm.sample_h

    sample_v = functools.partial(grbm.sample_v, add_noise=add_reconstruction_noise)
    grad = functools.partial(grbm.neg_free_energy_grad, learn_sigma=learn_sigma)

    # Optimization objective.
    def f(params, i):
        return rbm.cd(params, batches[i,:,:], sample_h, sample_v, grad)

    # Optimize.
    learning_rate = 0.01
    # TODO: Needs updating to use BatchIterator.
    raise Exception, 'See TODO.'
    num_batches = batches.shape[0]
    return optimize.sgd(f, params, num_batches, epochs, learning_rate, use_momentum)


np.random.seed(32)
inputs = load_inputs()

def ex1(inputs):
    """
    Gaussian/Bernoulli RBM.
    
    Learns edge detector like filters although learning is quite slow.

    Momentum seems neccesary, without it this fails to learn anything.
    (This was using momentum=0.5 for the first 5 epochs and
    momentum=0.9 thereafter.)

    Learning does happen if you don't use whitening, but the features
    tend to be less localized when compared to the learned with
    whitening. Interestingly the reconstruction error is lower without
    whitening, but I suspect I'm comparing apples with oranges there.

    With only 25 hidden units I couldn't find a way to learn anything
    much. Contrast this to an autoencoder which does seem to learn
    filters in a similar situation.

    Final error = 24.0968
    See ex1.png.
    """
    inputs = inputs[0:10000,:]
    inputs = utils.remove_dc(inputs)
    inputs, zca = utils.zca_white(inputs, 0.1)
    batches = optimize.make_batches(inputs, 50)
    num_vis = 64
    num_hid = 100
    epochs = 500
    initial_params = grbm.initial_params(num_vis, num_hid)
    params = train_rbm(batches, initial_params, epochs,
                       use_noisy_relu=False,
                       learn_sigma=False,
                       add_reconstruction_noise=False,
                       use_momentum=True)
    clf()
    v = 1.0
    imshow(utils.tile(params.W), vmin=-v, vmax=v)
    return params

def ex2(inputs):
    """
    Gaussian/NReLU RBM.

    Using noisy rectified linear units for the visibles speeds up
    learning dramatically. The reconstruction error after a single
    epoch is lower (21.6986) than after 500 epochs in ex1.

    The filters learned have less noisy backgrounds than those learned
    in ex1.

    Final error = 15.9089
    See ex2.png.
    """
    inputs = inputs[0:10000,:]
    inputs = utils.remove_dc(inputs)
    inputs, zca = utils.zca_white(inputs, 0.1)
    batches = optimize.make_batches(inputs, 50)
    num_vis = 64
    num_hid = 100
    epochs = 500
    initial_params = grbm.initial_params(num_vis, num_hid)
    params = train_rbm(batches, initial_params, epochs,
                       use_noisy_relu=True,
                       learn_sigma=False,
                       add_reconstruction_noise=False,
                       use_momentum=True)
    clf()
    v = 0.6
    imshow(utils.tile(params.W), vmin=-v, vmax=v)
    return params

def ex3(inputs):
    """
    Gaussian/NReLU RBM with learned visible variances.

    I found it *essential* to add noise/sample from the visible units
    during reconstruction. If I don't do this the variances increase
    at each epoch (I'd expect them to decrease during learning from
    their initial value of one) as does the error.
    
    This result was obtained by running SGD *without* momentum. The
    default momentum schedule set-up a big oscillation which caused
    the error to increase over a few epochs after which we learning
    appeared to be stuck out on a plateau. More modest schedules (such
    as 0.1 for the first 10 epochs, 0.2 thereafter) allow learning but
    they don't result in any improvement in error.
    
    The variances learned are all very similar. Their mean is 0.39,
    their variance is 0.04.

    Final error = 17.0089
    See ex3.png.
    """
    inputs = inputs[0:10000,:]
    inputs = utils.remove_dc(inputs)
    inputs, zca = utils.zca_white(inputs, 0.1)
    batches = optimize.make_batches(inputs, 50)
    num_vis = 64
    num_hid = 100
    epochs = 500
    initial_params = grbm.initial_params(num_vis, num_hid)
    params = train_rbm(batches, initial_params, epochs,
                       use_noisy_relu=True,
                       learn_sigma=True,
                       add_reconstruction_noise=True,
                       use_momentum=False)
    clf()
    v = 0.6
    imshow(utils.tile(params.W), vmin=-v, vmax=v)
    return params

params = ex3(inputs)
