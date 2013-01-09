import operator
import collections
import functools
import itertools
import numpy as np
import utils
from sigmoid import logistic

params = collections.namedtuple('params', 'W v_bias h_bias')

def initial_params(num_hid, num_vis):
    W = 0.05 * np.random.randn(num_hid, num_vis)
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
    h, h_mean = sample_hiddens(params, v, False)
    yield (v, h, h_mean)

    for i in xrange(length-1):
        v = sample_visibles(params, h, h_mean)
        h, h_mean = sample_hiddens(params, v, i==length-2)
        # Use a namedtuple here perhaps?
        yield (v, h, h_mean)

def sample_h(rbm, v, end_of_chain):
    h_mean = logistic(v.dot(rbm.W.T) + rbm.h_bias)
    if not end_of_chain:
        h = h_mean > np.random.random(h_mean.shape)
    else:
        # TODO: Can I return None to reduce ambiguity?
        h = None#h_mean
    return h, h_mean

def sample_v(rbm, h, h_mean):
    # Note that I'm currently *NOT* sampling the visibles. This may
    # make sense in some situations (perhaps cd1 and/or mnist) but in
    # general it is not correct.
    v_mean = logistic(h.dot(rbm.W) + rbm.v_bias)
    v = v_mean > np.random.random(v_mean.shape)
    return v

def neg_free_energy_grad(rbm, sample):
    v, h, h_mean = sample
    num_cases = v.shape[0]
    W_grad = h_mean.T.dot(v) / num_cases
    v_bias_grad = np.mean(v, 0)
    h_bias_grad = np.mean(h_mean, 0)
    return W_grad, v_bias_grad, h_bias_grad

def cd(params, data, sample_h, sample_v, neg_free_energy_grad,
       weight_penalty=0, k=1):
    # TODO: Have my optimize method work for acscent too. CD is
    # usually formulated as gradient ascent on the log probability, so
    # this change will make things a little easier to understand.
    
    # Could pass the gibs_chain function with sample_h and sample_v
    # applied (functools.partial) rather than passing in sample_h and
    # sample_v.

    gc = gibbs_chain(data, params, sample_h, sample_v, k+1)

    v0, h0, h0_mean = gc.next()
    v1, h1, h1_mean = itertools.islice(gc, k-1, None).next()

    pos_grad = neg_free_energy_grad(params, (v0, h0, h0_mean))
    neg_grad = neg_free_energy_grad(params, (v1, h1, h1_mean))

    # These is reversed (relative to way you see it written normally)
    # because my optimization functions does gradient *descent*.
    grad = map(operator.sub, neg_grad, pos_grad)

    # TODO: I've added this quickly, think and be sure it's the right
    # place for it.
    # TODO: Extract weight penalties to functions.
    # TODO: Do the terms weight cost and weight penalties make sense
    # as I use them?
    # L2 weight penalty.
    if weight_penalty > 0:
        grad = tuple(g_i + weight_penalty * (p_i.ndim - 1) * p_i
                     for p_i, g_i
                     in zip(params, grad))

    # Reconstruction error.
    num_cases = data.shape[0]
    error = np.sum((v0 - v1) ** 2) / num_cases

    return error, grad

def fantasize(gibbs_chain, start=None, step=None, count=None):
    """
    Convinience function to generate a stream of fantasies from an
    RBM.
    """
    g = itertools.islice(gibbs_chain, start, None, step)
    if count is not None:
        g = itertools.islice(g, count)
    # We need only the state of the visibles.
    g = itertools.imap(operator.itemgetter(0), g)
    g = itertools.imap(utils.tile, g)
    return g
