import operator
import collections
import functools
import itertools
import numpy as np
import utils
from sigmoid import logistic

params = collections.namedtuple('params', 'W v_bias h_bias')

def initial_params(num_hid, num_vis, sigma=0.01):
    # neg_free_energy_grad uses h_mean rather than h so small initial
    # weights are required to break the symmetry between hidden units.
    # Initializing from a normal with standard deviation of 0.01 is
    # recommended in "A Practical Guide to Training Restricted
    # Boltzmann Machines" and does appear to work well although
    # learning may initially be slow if testing with num_hid << 100.
    W = sigma * np.random.randn(num_hid, num_vis)
    v_bias = np.zeros(num_vis)
    h_bias = np.zeros(num_hid)
    return params(W, v_bias, h_bias)

MAX_CHAIN_LENGTH = 2**32

def gibbs_chain(v, params,
                sample_hiddens, sample_visibles,
                length=MAX_CHAIN_LENGTH):
    """
    Perform block Gibbs sampling.
    """
    assert length > 0

    # The first sample starts with the states of visible units given
    # by v.
    h, h_mean = sample_hiddens(params, v, length == 1)
    # We didn't compute v here, so we don't have v_mean to include in
    # this sample.
    yield (v, None, h, h_mean)

    for i in xrange(length - 1):
        v, v_mean = sample_visibles(params, h)
        h, h_mean = sample_hiddens(params, v, i == length - 2)
        p_new = (yield (v, v_mean, h, h_mean))
        if p_new is not None:
            params = p_new

# TODO: Rename end_of_chain to something like perform_sampling.
def sample_h(rbm, v, end_of_chain):
    h_mean = logistic(v.dot(rbm.W.T) + rbm.h_bias)
    if not end_of_chain:
        h = sample_bernoulli(h_mean)
    else:
        # Don't sample the states of the hidden units, because:
        # a) We're at the end of the gibbs chain so we don't need h
        # for future computations of p(v|h).
        # b) h isn't required to compute neg_free_energy_grad.
        h = None
    return h, h_mean

def sample_v(rbm, h):
    v_mean = logistic(h.dot(rbm.W) + rbm.v_bias)
    v = sample_bernoulli(v_mean)
    return v, v_mean

def sample_v_softmax(rbm, h, k, labels=None):
    """
    Sample the visible units of an RBM treating the k left-most units
    as a softmax group. If labels is given, the softmax group is
    clamped to those values.
    """
    # Top down activity for all units.
    a = h.dot(rbm.W) + rbm.v_bias
    # Softmax units.
    if labels is None:
        u = a[:,0:k] # Activities are un-normalized log probabilities.
        log_prob = u - np.log(np.sum(np.exp(u), 1))[:,np.newaxis]
        prob = np.exp(log_prob)
        labels = sample_softmax(prob)
    else:
        # Use labels as probs if clamped.
        prob = labels
    # Logistic units.
    v_mean = logistic(a[:,k:])
    v = sample_bernoulli(v_mean)
    return np.hstack((labels, v)), np.hstack((prob, v_mean))

def sample_softmax(p):
    num_cases = p.shape[0]
    out = np.zeros(p.shape)
    # TODO: Vectorize.
    # Something like would give return indexes:
    # (np.random.random((probs.shape[0], 1)) < np.cumsum(probs,
    # 1)).argmax(1)
    # Would just need to turn those back into 1-of-n vectors.
    # (Probably by indexing the identity function.) If implemented,
    # test it is actually faster.
    for i in xrange(num_cases):
        out[i] = np.random.multinomial(1, p[i], 1)
    return out

# TODO: Extract from this modules, useful elsewhere. e.g. DBN fine tuning.
def sample_bernoulli(p):
    # Python will implicitly convert True/False to 1/0, but I'm doing
    # it explicitly for clarity. e.g. When inspecting the returned
    # array during debugging.
    return (p > np.random.random(p.shape)) * 1.

def neg_free_energy_grad(rbm, sample):
    v, v_mean, h, h_mean = sample
    num_cases = v.shape[0]
    W_grad = h_mean.T.dot(v) / num_cases
    v_bias_grad = np.mean(v, 0)
    h_bias_grad = np.mean(h_mean, 0)
    return W_grad, v_bias_grad, h_bias_grad

def cd(params, data, sample_h, sample_v, neg_free_energy_grad,
       weight_decay=None, k=1):
    # For PCD it's not straight forward to re-use the same chain as
    # the params are changing over time but the chain is stuck with
    # the params it was created with.
    gc = gibbs_chain(data, params, sample_h, sample_v, k + 1)

    # Positive sample.
    pos_sample = gc.next()
    # Sample used to compute one-step reconstruction error.
    recon_sample = gc.next()
    # Negative sample.
    if k == 1:
        neg_sample = recon_sample
    else:
        neg_sample = itertools.islice(gc, k - 2, None).next()

    pos_grad = neg_free_energy_grad(params, pos_sample)
    neg_grad = neg_free_energy_grad(params, neg_sample)

    # These is reversed (relative to way you see it written normally)
    # because my optimization functions does gradient *descent*.
    grad = map(operator.sub, neg_grad, pos_grad)

    if weight_decay:
        weight_grad = (weight_decay(p)[1] for p in params)
        grad = map(operator.add, grad, weight_grad)

    # Assuming grad is a list, wouldn't this be clearer. (And possibly
    # faster because it's in-place. (Which is particularly useful for
    # biases where we're probably adding 0.)
    # if weight_decay:
    #     for i, p in enumerate(params):
    #         grad[i] += weight_decay(p)[1]

    # One-step reconstruction error.
    num_cases = data.shape[0]
    error = np.sum((data - recon_sample[1]) ** 2) / num_cases

    return error, grad

def pcd(sample_h, sample_v, neg_free_energy_grad, weight_decay=None):
    def f(rbm, data):
        # I could use gibbs_chain to collect the positive phase
        # statistics but doing so would perform an unnecessary upward
        # pass. (From the reconstruction to the hidden units.)
        h, h_mean = sample_h(rbm, data, False)

        # Reconstruction, used cost approximation.
        v, v_mean = sample_v(rbm, h)

        # neg_free_energy_grad takes a sample as an argument, so
        # construct one.
        pos_sample = (data, None, h, h_mean)

        if f.gc is None:
            # Initialize the persistent chain.
            # The visible states from which to start the Gibbs chain.
            # I have no idea whether this strategy is optimal.
            initial_v = np.random.random(data.shape)
            f.gc = gibbs_chain(initial_v, rbm, sample_h, sample_v)
            neg_sample = f.gc.next()
        else:
            # Get the negative samples from the persistent gibbs chain
            # using the new parameters.
            neg_sample = f.gc.send(rbm)

        # Compute the gradient.
        pos_grad = neg_free_energy_grad(rbm, pos_sample)
        neg_grad = neg_free_energy_grad(rbm, neg_sample)
        grad = map(operator.sub, neg_grad, pos_grad)

        if weight_decay:
            weight_grad = (weight_decay(p)[1] for p in rbm)
            grad = map(operator.add, grad, weight_grad)

        # One-step reconstruction error.
        num_cases = data.shape[0]
        error = np.sum((data - v_mean) ** 2) / num_cases

        # Keep a running average of the mean hidden unit activation
        # probability.
        _lambda = 0.9
        f.q = _lambda * f.q + (1 - _lambda) * h_mean.mean()

        return error, grad

    f.gc = None
    f.q = 0
    return f
