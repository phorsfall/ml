import collections

from ml import rbm

layer = collections.namedtuple('layer', 'W_r b_r W_g b_g')

def stack_params(params):
    """
    Stack a single tuple of DBN parameters layer-wise as a list.
    """
    # The general idea here is that by having the parameters arrange
    # in this way it will be easier to do things like iterate over
    # layers as some point. The indexing would be awkward if I tried
    # to do this with all the parameters in a single tuple. (Which is
    # what all of my optimization/regularization code expects.)

    # The last 3 items of params form the top-level RBM. The remaining
    # items for the lower layers, with each layer specified by 4
    # items.
    num_layers = (len(params) - 3) / 4
    stack = []
    for i in range(num_layers):
        stack.append(layer._make(params[4*i:4*(i+1)]))
    stack.append(rbm.params._make(params[-3:]))
    return stack

def params_from_rbms(rbms):
    """
    Combine the parameters from a list of RBMs into a single tuple.
    Recongnition and generative weights are un-tied. The <rbms>
    parameter should be a list of RBMs in layer order starting with
    the loest (visible) layer and ending with the top-level
    (associative memory) RBM.
    """
    params = []
    for rbm in rbms[:-1]:
        # W_r, b_r, W_g, b_g
        params.extend([rbm.W.T.copy(), rbm.h_bias, rbm.W, rbm.v_bias])
    params.extend(rbms[-1])
    return tuple(params)
