import collections
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

def cost(params, inputs, targets, weight_penalty=0.):
    num_cases = inputs.shape[0]
    
    # Un-normalized log-prob.
    U = inputs.dot(params.W.T) + params.b
    # Normalize.
    log_prob = U - np.log(np.sum(np.exp(U), 1))[:,np.newaxis]
    # Compute probabilities over classes.
    prob = np.exp(log_prob)

    # np.dot(a, a) is the idiomatic way to do sum of squares, and is
    # faster than np.sum(a ** 2). For 2d arrays even b = a.ravel();
    # np.dot(b, b) appears quicker.
    weight_cost = (0.5 * weight_penalty * (
            np.sum(params.W ** 2) + np.sum(params.b ** 2)))

    cost = weight_cost - (np.sum(log_prob * targets) / num_cases)

    error = prob - targets
    W_grad = ((error).T.dot(inputs) / num_cases) + (weight_penalty * params.W)
    b_grad = (np.sum(error, 0) / num_cases) + (weight_penalty * params.b)

    return cost, (W_grad, b_grad)

def predict(inputs, params):
    U = inputs.dot(params.W.T) + params.b
    log_prob = U - np.log(np.sum(np.exp(U), 1))[:,np.newaxis]
    prob = np.exp(log_prob)
    return np.argmax(prob, 1)

def accuracy(inputs, labels, params):
    return np.mean(predict(inputs, params) == labels) * 100
