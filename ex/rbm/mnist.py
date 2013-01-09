import os.path
import functools
import tempfile

import numpy as np

from ml import mnist, utils, optimize, rbm, data

print '***eval***'

DATA_PATH = os.path.expanduser('~/Development/ml/datasets')
OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

np.random.seed(32)

inputs =  mnist.load_inputs(DATA_PATH, mnist.TRAIN_INPUTS)
batches = data.BatchIterator(inputs)
num_vis = inputs.shape[1]
num_hid = 500
epochs = 25
k = 3 # cd_k
initial_params = rbm.initial_params(num_hid, num_vis)
weight_constraint = None
weight_penalty = 0
momentum = optimize.linear(0.5, 0.9, 10)
# momentum = 0
learning_rate = 0.1

def generate(params):
    random_pixels = np.random.random_sample((16, num_vis)) * 0.1
    gc = rbm.gibbs_chain(random_pixels,
                         params,
                         rbm.sample_h,
                         rbm.sample_v)

    fantasies = rbm.fantasize(gc, 0, 200, 20)
    utils.save_images(fantasies, tempfile.mkdtemp(dir=OUTPUT_PATH))

def f(params, inputs):
    return rbm.cd(params, inputs,
                  rbm.sample_h,
                  rbm.sample_v,
                  rbm.neg_free_energy_grad,
                  weight_penalty=weight_penalty,
                  k=k)

save_params = utils.save_params_hook(OUTPUT_PATH)

def post_epoch(*args):
    save_params(*args)

params = optimize.sgd(f, initial_params, batches, epochs,
                      learning_rate,
                      momentum,
                      weight_constraint=weight_constraint,
                      post_epoch=post_epoch)
