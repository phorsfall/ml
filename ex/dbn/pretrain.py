import os.path
import functools

import numpy as np

from ml import mnist, utils, optimize, data, rbm, meta, parameters
from ml import regularization as reg
from ml.sigmoid import logistic

DATA_PATH = os.path.expanduser('~/Development/ml/datasets')
OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

np.random.seed(32)

sample_v_softmax = functools.partial(rbm.sample_v_softmax, k=mnist.NUM_CLASSES)    

def sgd(f, initial_params, batches, momentum, hook=None):
    output_dir = utils.make_output_directory(OUTPUT_PATH)
    save_params = parameters.save_hook(output_dir)
    def post_epoch(*args):
        save_params(*args)
        if hook is not None:
            hook(*args)
    return optimize.sgd(f, initial_params, batches,
                        epochs, learning_rate, momentum,
                        post_epoch=post_epoch)

# Optimization objective for the first two layers.
def rbm_obj(params, inputs):
    return rbm.cd(params, inputs,
                  rbm.sample_h,
                  rbm.sample_v,
                  rbm.neg_free_energy_grad,
                  weight_decay)

epochs = 100
learning_rate = 0.1
weight_decay = reg.l2(0.0002)

momentum = meta.linear(0.5, 0.9, 10)

inputs = mnist.load_inputs(DATA_PATH, mnist.TRAIN_INPUTS)
labels = mnist.load_labels(DATA_PATH, mnist.TRAIN_LABELS)
inputs, targets, labels = data.balance_classes(inputs, labels, mnist.NUM_CLASSES)
n = 54200
inputs = inputs[0:n]
targets = targets[0:n]
labels = labels[0:n]

# These layers differ slightly from those in the paper. My main
# motivation is to avoid having a square weight matrix between hidden
# layers to avoid matrix transpose errors.
num_vis = inputs.shape[1]
num_hid1 = 529 # 23^2 
num_hid2 = 484 # 22^2
num_top = 1936 # 44^2

batches = data.BatchIterator(inputs)
initial_params = rbm.initial_params(num_hid1, num_vis)
params = sgd(rbm_obj, initial_params, batches, momentum)

inputs = logistic(inputs.dot(params.W.T) + params.h_bias)
batches = data.BatchIterator(inputs)
initial_params = rbm.initial_params(num_hid2, num_hid1)
params = sgd(rbm_obj, initial_params, batches, momentum)

inputs = logistic(inputs.dot(params.W.T) + params.h_bias)
batches = data.BatchIterator(np.hstack((targets, inputs)))
initial_params = rbm.initial_params(num_top, num_hid2 + mnist.NUM_CLASSES)

def post_epoch(*args):
    print 'Mean hidden activation prob. is %.2f' % pcd.q

# Optimization objective for the top-level RBM.
pcd = rbm.pcd(rbm.sample_h, sample_v_softmax,
              rbm.neg_free_energy_grad, weight_decay)

params = sgd(pcd, initial_params, batches, 0, post_epoch)
