import collections
import numpy as np
from sigmoid import logistic

params = collections.namedtuple('params', 'W v_bias h_bias sigma')

def initial_params(num_vis, num_hid):
    W = 0.05 * np.random.randn(num_hid, num_vis)
    v_bias = np.zeros(num_vis)
    h_bias = np.zeros(num_hid)
    sigma = np.ones(num_vis)
    return params(W, v_bias, h_bias, sigma)

def sample_h(rbm, v, end_of_chain):
    h_mean = logistic((v / rbm.sigma).dot(rbm.W.T) + rbm.h_bias)
    if not end_of_chain:
        h = h_mean > np.random.random(h_mean.shape)
    else:
        h = None
    return h, h_mean

def sample_h_noisy_relu(rbm, v, end_of_chain):
    propup = (v / rbm.sigma).dot(rbm.W.T) + rbm.h_bias
    h_mean = np.maximum(0, propup)
    if not end_of_chain:
        noise = np.sqrt(logistic(propup)) * np.random.standard_normal(propup.shape)
        h = np.maximum(0, propup + noise)
    else:
        h = None
    return h, h_mean    

def sample_v(rbm, h, h_mean, add_noise=True):
    v_mean = (h.dot(rbm.W) * rbm.sigma) + rbm.v_bias
    # I'm considering creating a separate sample_v_with_noise
    # function rather than having this conditional.
    if add_noise:
        v = rbm.sigma * np.random.standard_normal(v_mean.shape) + v_mean
    else:
        v = v_mean
    return v

def neg_free_energy_grad(rbm, sample, learn_sigma=True):
    v, h, h_mean = sample
    num_cases, num_vis = v.shape
    
    W_grad = h_mean.T.dot(v / rbm.sigma) / num_cases
    v_bias_grad = np.mean((v - rbm.v_bias) / (rbm.sigma**2), 0)
    h_bias_grad = np.mean(h_mean, 0)
    
    if learn_sigma:
        sigma_term_1 = ((v - rbm.v_bias) ** 2) / (rbm.sigma**3)
        sigma_term_2 = (v / (rbm.sigma**2)) * h_mean.dot(rbm.W)
        sigma_grad = np.mean(sigma_term_1 - sigma_term_2, 0)
    else:
        sigma_grad = np.zeros(num_vis)

    return W_grad, v_bias_grad, h_bias_grad, sigma_grad
