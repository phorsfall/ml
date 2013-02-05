import os.path
import cPickle as pickle
import functools

import numpy as np
from PIL import Image

from ml import mnist, utils, optimize, rbm, grbm, data, parameters
from ml import regularization as reg

DATA_PATH = os.path.expanduser("~/Development/ml/datasets")
OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

# A visualization of the weights learned by this RBM after 100 epochs
# is here:
# https://dl.dropbox.com/u/501760/ml/grbm-color-patches.png
# (Far fewer epochs were nessecary to learn visually similar weights.)

def load_inputs():
    fn = 'natural-image-patches-color-100k-16x16.pickle'
    with(open(os.path.join(DATA_PATH, fn))) as f:
        data = pickle.load(f)
    print data['desc']
    return data['inputs']

def zero_mean(X):
    return X - X.mean(0)

def ex(inputs):
    inputs = zero_mean(inputs)
    inputs, zca = utils.zca_white(inputs, 0.1)
    batches = data.BatchIterator(inputs, 100)
    num_vis = inputs.shape[1]
    num_hid = 400
    epochs = 100
    momentum = 0

    initial_params = grbm.initial_params(num_hid, num_vis, 0.001, 0.4)

    neg_free_energy_grad = functools.partial(grbm.neg_free_energy_grad,
                                             learn_sigma=False)

    def f(params, inputs):
        return rbm.cd(params, inputs,
                      grbm.sample_h_noisy_relu, grbm.sample_v,
                      neg_free_energy_grad)
    
    learning_rate = 0.005

    output_dir = utils.make_output_directory(OUTPUT_PATH)
    save_params = parameters.save_hook(output_dir)

    def post_epoch(*args):
        save_params(*args)
        # Save visualization weights.
        W_norm = utils.rescale(args[0].W)
        img = Image.fromarray(np.uint8(utils.tile(W_norm, channel_count=3) * 255))
        img.save(os.path.join(output_dir, ('w%i.png' % args[1])))
        # Estimate sparsity from subset of data.
        h_mean = grbm.sample_h_noisy_relu(args[0], inputs[0:5000], True)[1]
        mean_activation = np.mean(h_mean > 0)
        print 'approx mean activation: %f' % mean_activation

    return optimize.sgd(f, initial_params, batches,
                        epochs, learning_rate,
                        momentum,
                        post_epoch=post_epoch)

np.random.seed(32)
inputs = load_inputs()
params = ex(inputs)
