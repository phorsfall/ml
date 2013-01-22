import os.path
import functools
import itertools
import tempfile
import operator

import numpy as np

from ml import mnist, utils, optimize, rbm, data, parameters, meta
from ml import regularization as reg

print '***eval***'

DATA_PATH = os.path.expanduser('~/Development/ml/datasets')
OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

np.random.seed(32)

# data
inputs =  mnist.load_inputs(DATA_PATH, mnist.TRAIN_INPUTS)
labels = mnist.load_labels(DATA_PATH, mnist.TRAIN_LABELS)
inputs = data.balance_classes(inputs, labels, mnist.NUM_CLASSES).inputs
batches = data.BatchIterator(inputs[0:54200])

# IDEA: Have a bit of code which loads the current file (as text) and
# dumps and code between two specific comments into a meta.txt file in
# the output directory.

# meta
num_vis = inputs.shape[1]
num_hid = 49
epochs = 10
k = 1 # applies to cd only
initial_params = rbm.initial_params(num_hid, num_vis)
weight_constraint = None
# weight_decay = reg.l2(0.0002)
# weight_decay = None
# momentum = meta.linear(0.5, 0.9, 10)
momentum = 0
learning_rate = 0.1

def generate(params, n=1, start=1, step=20, count=100):
    # Use the probability as pixel intensities.
    # This means we can't use the very first sample from the chain, as
    # it has v_mean == None.
    assert start > 0

    num_vis = params.W.shape[1]
    #initial_v = np.zeros((n, num_vis))
    initial_v = inputs[:n]
    #initial_v = np.random.random((n, num_vis))
    gc = rbm.gibbs_chain(initial_v,
                         params,
                         rbm.sample_h,
                         rbm.sample_v)

    g = itertools.islice(gc, start, None, step)
    g = itertools.islice(g, count)
    g = itertools.imap(operator.itemgetter(1), g)
    g = itertools.imap(utils.tile, g)
    utils.save_images(g, tempfile.mkdtemp(dir=OUTPUT_PATH))

# def f(params, inputs):
#     return rbm.cd(params, inputs,
#                   rbm.sample_h,
#                   rbm.sample_v,
#                   rbm.neg_free_energy_grad,
#                   weight_decay=weight_decay,
#                   k=k)


output_dir = utils.make_output_directory(OUTPUT_PATH)
save_params = parameters.save_hook(output_dir)

f = rbm.pcd(rbm.sample_h, rbm.sample_v,
            rbm.neg_free_energy_grad, weight_decay)

def post_epoch(*args):
    #save_params(*args)
    #print 'Mean hidden activation prob. is %f.' % f.q
    pass
   
params = optimize.sgd(f, initial_params, batches, epochs,
                      learning_rate,
                      momentum,
                      weight_constraint=weight_constraint,
                      post_epoch=post_epoch)
