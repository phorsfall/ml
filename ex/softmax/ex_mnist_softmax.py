import os

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'none'

from pylab import *
ion()

import numpy as np

import softmax
import optimize
import utils
import mnist

reload(softmax)
reload(optimize)

print '*** eval ***'

def train(params, batches, weight_penalty,
          epochs, learning_rate,
          callback):
    def f(params, inputs, targets):
        return softmax.cost(params, inputs, targets, weight_penalty)
    return optimize.sgd(f, params, batches, epochs, learning_rate,
                        False, post_epoch=callback)

# I'm not 100% sure I like this way of handling this.
def make_accuracy_callback(train_inputs, train_labels,
                           test_inputs, test_labels,
                           acc_func):
    def callback(cost, params):
        train_acc = acc_func(train_inputs, train_labels, params)
        test_acc = acc_func(test_inputs, test_labels, params)
        callback.train_history.append(train_acc)
        callback.test_history.append(test_acc)
    callback.train_history = []
    callback.test_history = []
    return callback

DATA_PATH = os.path.expanduser('~/Datasets')

# TODO: Have a named tuple to hold inputs/labels/targets might be
# convinient.

#inputs = mnist.load_inputs(DATA_PATH, mnist.TRAIN_INPUTS)/255.
#labels = mnist.load_labels(DATA_PATH, mnist.TRAIN_LABELS)

#test_inputs = mnist.load_inputs(DATA_PATH, mnist.TEST_INPUTS)/255.
#test_labels = mnist.load_labels(DATA_PATH, mnist.TEST_LABELS)
num_classes = 10
num_dims = inputs.shape[1]

targets = utils.targets_from_labels(labels, num_classes)
test_targets = utils.targets_from_labels(test_labels, num_classes)

batches = optimize.make_batches(inputs, targets)

initial_params = softmax.initial_params(num_classes, num_dims)

weight_penalty = 1e-4
epochs = 25
learning_rate = 0.1

# def f(params):
#     return softmax.cost(params,
#                         inputs[0:2,:],
#                         targets[0:2,:],
#                         1e-3)

# print optimize.check_gradient(f, initial_params)



callback = make_accuracy_callback(inputs, labels,
                                  test_inputs, test_labels,
                                  softmax.accuracy)

params = train(initial_params,
               batches,
               weight_penalty,
               epochs,
               learning_rate,
               callback)

accuracy = softmax.accuracy(inputs, labels, params)
print 'Training accuracy: %.2f%%' % accuracy

accuracy = softmax.accuracy(test_inputs, test_labels, params)
print 'Test accuracy: %.2f%%' % accuracy

#clf()
#imshow(utils.tile(params.W, (2, 5)))

