import os.path
import cPickle as pickle
import functools

import numpy as np

from ml import mnist, utils, optimize, rbm, grbm, data, meta, parameters
from ml import regularization as reg

DATA_PATH = os.path.expanduser("~/Development/ml/datasets")
OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

def load_inputs():
    # This data set is sampled from Bruno Olshausen's unwhitened
    # natural images: http://redwood.berkeley.edu/bruno/sparsenet/
    # Send me an email (see GitHub profile page) if you'd like a copy
    # of the exact patches I used here.
    fn = 'natural-image-patches-100k-12x12.pickle'
    with(open(os.path.join(DATA_PATH, fn))) as f:
        data = pickle.load(f)
    print data['desc']
    return data['inputs']

def ex(inputs):
    inputs = utils.remove_dc(inputs)
    inputs, zca = utils.zca_white(inputs, 0.1)
    batches = data.BatchIterator(inputs, 100)
    num_vis = inputs.shape[1]
    num_hid = 400
    epochs = 100
    momentum = 0

    initial_params = grbm.initial_params(num_hid, num_vis, 0.001, 1.0)

    neg_free_energy_grad = functools.partial(grbm.neg_free_energy_grad,
                                             learn_sigma=False)

    def f(params, inputs):
        return rbm.cd(params, inputs,
                      grbm.sample_h_noisy_relu, grbm.sample_v,
                      neg_free_energy_grad)
    
    learning_rate = 0.005

    output_dir = utils.make_output_directory(OUTPUT_PATH)
    save_params = parameters.save_hook(output_dir)
    error_history = []
    sparsity_history = []

    def post_epoch(*args):
        W_norm = utils.rescale(args[0].W)
        utils.save_image(utils.tile(W_norm),
                         os.path.join(output_dir, ('w%i.png' % args[1])))

        # Estimate sparsity from subset of data.
        h_mean = grbm.sample_h_noisy_relu(args[0], inputs[0:5000], True)[1]
        mean_activation = np.mean(h_mean > 0)
        print 'approx mean activation: %f' % mean_activation
        
        # The callback from optimize.sgd needs modifying so that it
        # passes the reconstrcution error as an argument to make this
        # work. (This was used when I did the original experiments.)
        # error_history.append(args[2])
        sparsity_history.append(mean_activation)
        
        save_params(args[0], args[1])

    params = optimize.sgd(f, initial_params, batches,
                          epochs, learning_rate,
                          momentum,
                          post_epoch=post_epoch)

    with(open(os.path.join(output_dir, 'history.pickle'), 'wb')) as f:
        pickle.dump(error_history, f, -1)
        pickle.dump(sparsity_history, f, -1)

    return params, error_history, sparsity_history

np.random.seed(32)
inputs = load_inputs()
params, error_history, sparsity_history = ex(inputs)
