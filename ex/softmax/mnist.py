import os.path
import functools
import numpy as np
from ml import softmax, optimize, utils, mnist, data, meta
from ml import regularization as reg

DATA_PATH = os.path.expanduser('~/Development/ml/datasets')

train, valid = mnist.load_training_dataset(DATA_PATH)
batches = data.BatchIterator(train)

num_classes = mnist.NUM_CLASSES
num_dims = train.inputs.shape[1]

initial_params = softmax.initial_params(num_classes, num_dims)

weight_decay = None # reg.l2(1e-4)
epochs = 50
learning_rate = 0.1
momentum = 0

def f(params, data):
    return softmax.cost(params, data.inputs, data.targets, weight_decay)

train_accuracy = functools.partial(softmax.accuracy,
                                   train.inputs, train.labels)

valid_accuracy = functools.partial(softmax.accuracy,
                                   valid.inputs, valid.labels)

train_error = utils.call_func_hook(train_accuracy)
valid_error = utils.call_func_hook(valid_accuracy)

def post_epoch(*args):
    train_error(*args)
    valid_error(*args)

params = optimize.sgd(f, initial_params, batches,
                      epochs, learning_rate,
                      momentum=momentum,
                      post_epoch=post_epoch)
