import cPickle as pickle
import collections
import os

OUTPUT_PATH = os.path.expanduser('~/Development/ml/output')

# Keeping this file around for reference.

def load():
    with open(os.path.join(OUTPUT_PATH, 'mnist_dbn_1.pickle'), 'rb') as f:
        params1 = pickle.load(f)
    with open(os.path.join(OUTPUT_PATH, 'mnist_dbn_2.pickle'), 'rb') as f:
        params2 = pickle.load(f)
    with open(os.path.join(OUTPUT_PATH, 'mnist_dbn_3.pickle'), 'rb') as f:
        params3 = pickle.load(f)
    return params1, params2, params3

p = load()

W_r1 = p[0].W.copy()
W_g1 = p[0].W.copy()

b_r1 = p[0].h_bias
b_g1 = p[0].v_bias

W_r2 = p[1].W.copy()
W_g2 = p[1].W.copy()

b_r2 = p[1].h_bias
b_g2 = p[1].v_bias

W = p[2].W
v_bias = p[2].v_bias
h_bias = p[2].h_bias

p_tuple = (W_r1, b_r1, W_r2, b_r2, W_g1, b_g1, W_g2, b_g2, W, v_bias, h_bias)

# Combined parameters saved as a plain tuple to ensure it's easy to
# unpickle.
with(open(os.path.join(OUTPUT_PATH, 'mnist_dbn_pretrain.pickle'), 'wb')) as f:
    pickle.dump(p_tuple, f, -1)
