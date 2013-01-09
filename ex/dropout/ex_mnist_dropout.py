import os.path

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'none'

from pylab import *
ion()

import numpy as np

import feedforward
import optimize
import utils
import mnist
import data

reload(feedforward)
reload(optimize)
reload(utils)
#reload(mnist)
#reload(data)

DATA_PATH = os.path.expanduser('~/Development/ml/datasets')
OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

training, validation = mnist.load_training_dataset(DATA_PATH)
batches = data.BatchIterator(training)

# These parameters (which match the experiment with 800 hidden units
# from the dropout paper) gave a classification error of 1.33% on the
# official test set. I trained on the first 50,000 examples and
# selected parameters with the lowest cross-entropy error on the
# validation set. (These were from epoch 963.) Training started at
# 1357064417 and took about 2 days on a CPU. (My implementation is not
# all that efficient.) This test set error is higher than the one
# reported in the paper. My best guess as to why this might be, is
# that I only trained on 50,000 cases and the authors may have
# re-trained on all 60,000 cases after selected meta-parameters. (The
# paper doesn't describe this aspect of the experiment.)

# See the 'dropout' directory in the same directory as this file for
# more results.

# In a previous experiment I used fixed learning rate (of perhaps
# 0.1), momentum of 0.5 for the first 5 epochs and 0.9 thereafter, and
# hidden layers of size 784 and 256. I believe this achived very
# similar results to the experiment above (using the meta-parameters
# from the paper) in only 250 epochs, although I didn't check the test
# error.

num_vis = training.inputs.shape[1]
num_hid1 = 800
num_hid2 = 800
num_classes = mnist.NUM_CLASSES

weight_penalty = 0
epochs = 1000
momentum = optimize.linear(0.5, 0.99, 500)
decaying_rate = optimize.exponential(10.0, 0.998)
learning_rate = lambda t: (1 - momentum(t)) * decaying_rate(t)

dropout = (0.2, 0.5, 0.5)
weight_constraint = np.sqrt(15) # Note this param specifies the max
                                # length not max sq. length.

initial_params = feedforward.initial_params(num_vis, num_hid1, num_hid2, num_classes,
                                            feedforward.init_random_normal)

save_params = utils.save_params_hook(OUTPUT_PATH)
training_error = utils.call_func_hook(lambda params:
                                          feedforward.error(params,
                                                            training,
                                                            dropout))
validation_error = utils.call_func_hook(lambda params:
                                            feedforward.error(params,
                                                              validation,
                                                              dropout))

def post_epoch(*args):
    save_params(*args)
    # training_error(*args)
    validation_error(*args)

def f(params, data):
    return feedforward.cost(params, data,
                            weight_penalty, dropout)

params = optimize.sgd(f, initial_params, batches,
                      epochs, learning_rate,
                      momentum,
                      weight_constraint,
                      post_epoch=post_epoch)
