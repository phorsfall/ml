import collections
import operator
import numpy as np

params = collections.namedtuple('params', 'W b')

def initial_params(num_classes, num_dims):
    W = np.zeros((num_classes, num_dims))
    b = np.zeros(num_classes)
    return params(W, b)

# Not currently used, not sure I'll see over/under flow on MNIST
# either as inputs as scaled [0,1] and weights will be relatively
# small.
# I assume there's a fast/stable logsumexp in np.
def logsumexp(A):
    max_a = np.max(A, 1)
    return np.log(np.sum(np.exp(A - max_a), 1)) + max_a

def cost(params, inputs, targets, weight_decay=None):
    num_cases = inputs.shape[0]
    
    # Un-normalized log-prob.
    U = inputs.dot(params.W.T) + params.b
    # Normalize.
    log_prob = U - np.log(np.sum(np.exp(U), 1))[:,np.newaxis]
    # Compute probabilities over classes.
    prob = np.exp(log_prob)

    error = prob - targets
    W_grad = ((error).T.dot(inputs) / num_cases)
    b_grad = (np.sum(error, 0) / num_cases)

    cost = -(np.sum(log_prob * targets) / num_cases)
    grad = (W_grad, b_grad)

    if weight_decay:
        weight_grad = []
        for p in params:
            wc, wg = weight_decay(p)
            weight_grad.append(wg)
            cost += wc
        grad = map(operator.add, grad, weight_grad)

    return cost, grad

def predict(inputs, params):
    U = inputs.dot(params.W.T) + params.b
    log_prob = U - np.log(np.sum(np.exp(U), 1))[:,np.newaxis]
    prob = np.exp(log_prob)
    return np.argmax(prob, 1)

def output(params, inputs):
    U = inputs.dot(params.W.T) + params.b
    log_prob = U - np.log(np.sum(np.exp(U), 1))[:,np.newaxis]
    prob = np.exp(log_prob)
    return prob

def accuracy(inputs, labels, params):
    return np.mean(predict(inputs, params) == labels) * 100
