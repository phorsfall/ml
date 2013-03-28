import collections
import numpy as np
from scipy.stats import bernoulli
from sigmoid import logistic

# Initially implements a net with 2 hidden layers (hard-coded) and
# a softmax output layer.

params = collections.namedtuple('params', 'W1 b1 W2 b2 W3 b3')

def init_weights_uniform(shape):
    epsilon = 0.12
    return np.random.random(shape) * epsilon * 2 - epsilon

def init_random_normal(shape):
    # This is based on the approach in the dropout paper.
    sigma = 0.01 # std. dev.
    return np.random.standard_normal(shape) * sigma

def initial_params(num_vis, num_hid1, num_hid2, num_classes,
                   init_weights=init_weights_uniform):
    W1 = init_weights((num_hid1, num_vis))
    W2 = init_weights((num_hid2, num_hid1))
    W3 = np.zeros((num_classes, num_hid2))
    b1 = np.zeros(num_hid1)
    b2 = np.zeros(num_hid2)
    b3 = np.zeros(num_classes)
    return params(W1, b1, W2, b2, W3, b3)

DEFAULT_DROPOUT = (0, 0, 0)

def cost(params, data, weight_penalty=0, dropout=None, rng=np.random):
    inputs, targets = data.inputs, data.targets
    W1, b1, W2, b2, W3, b3 = params

    if dropout is None:
        dropout = DEFAULT_DROPOUT

    num_cases = inputs.shape[0]
    a1 = inputs

    if dropout[0] > 0:
        mask = rng.random_sample(a1.shape) > dropout[0]
        a1 = a1 * mask

    z2 = a1.dot(W1.T) + b1
    a2 = logistic(z2)

    # Note that at present every single activation is computed even
    # though we throw half of that work away when using dropout.
    if dropout[1] > 0:
        mask = rng.random_sample(a2.shape) > dropout[1]
        a2 = a2 * mask
 
    z3 = a2.dot(W2.T) + b2
    a3 = logistic(z3)

    if dropout[2] > 0:
        mask = rng.random_sample(a3.shape) > dropout[2]
        a3 = a3 * mask

    # Un-nomalized log-prob.
    U = a3.dot(W3.T) + b3
    # Normalize.
    log_prob = U - np.log(np.sum(np.exp(U), 1))[:,np.newaxis]
    # Compute probabilities over classes.
    prob = np.exp(log_prob)

    weight_cost = (0.5 * weight_penalty *  (
            np.sum(W1 ** 2) + np.sum(b1 ** 2) +
            np.sum(W2 ** 2) + np.sum(b2 ** 2) +
            np.sum(W3 ** 2) + np.sum(b3 ** 2)))

    cost = weight_cost - (np.sum(log_prob * targets) / num_cases)

    delta4 = error = prob - targets
    delta3 = delta4.dot(W3) * a3 * (1 - a3)
    delta2 = delta3.dot(W2) * a2 * (1 - a2)

    W1_grad = (delta2.T.dot(a1) / num_cases) + (weight_penalty * W1)
    W2_grad = (delta3.T.dot(a2) / num_cases) + (weight_penalty * W2)
    W3_grad = (delta4.T.dot(a3) / num_cases) + (weight_penalty * W3)

    b1_grad = (np.sum(delta2, 0) / num_cases) + (weight_penalty * b1)
    b2_grad = (np.sum(delta3, 0) / num_cases) + (weight_penalty * b2)
    b3_grad = (np.sum(delta4, 0) / num_cases) + (weight_penalty * b3)

    return cost, (W1_grad, b1_grad, W2_grad, b2_grad, W3_grad, b3_grad)

def log_prob(params, data, dropout=None):
    """
    Compute the log probability over classes for each input.
    """
    W1, b1, W2, b2, W3, b3 = params

    if dropout is None:
        dropout = DEFAULT_DROPOUT

    a1 = data.inputs
    z2 = a1.dot(W1.T * (1 - dropout[0])) + b1
    a2 = logistic(z2) 
    z3 = a2.dot(W2.T * (1 - dropout[1])) + b2
    a3 = logistic(z3)
    U = a3.dot(W3.T * (1 - dropout[2])) + b3
    log_prob = U - np.log(np.sum(np.exp(U), 1))[:,np.newaxis]
    return log_prob

def error(params, data, dropout=None):
    lp = log_prob(params, data, dropout)
    accuracy = np.mean(np.argmax(lp, 1) == data.labels)
    classification_error = 1 - accuracy
    num_cases = data.inputs.shape[0]
    cross_entropy_error = -np.sum(lp * data.targets) / num_cases
    return classification_error, cross_entropy_error
