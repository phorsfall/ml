import sys
import itertools
import operator
import time
import datetime

import numpy as np

def gradient_descent(f, x0, num_epochs, learning_rate):
    make_tuple = _make_tuple_func(x0)
    x = x0
    for i in xrange(num_epochs):
        cost, grad = f(x)
        x = make_tuple(x_i - learning_rate * grad_i
                       for x_i, grad_i
                       in zip(x, grad))
        print cost
    return x

def sgd(f, x0, batches, num_epochs,
        learning_rate,
        momentum=0.,
        weight_constraint=None,
        post_epoch=None,
        rmsprop=False):
    """
    Perform stochastic gradient descent on the function f.
    """
    make_tuple = _make_tuple_func(x0)
    x = x0
    momentum = _wrap(momentum)
    learning_rate = _wrap(learning_rate)
    t0 = time.time()

    # Initialize parameter delta. This is maintained outside of the
    # main loop for purpose of implementing momentum.
    delta = [np.zeros_like(x_i) for x_i in x]
    
    # Running average for rmsprop.
    avg_sq_grad = [np.ones_like(x_i) for x_i in x]

    print

    for epoch in xrange(num_epochs):
        cost = 0

        p = momentum(epoch)
        e = learning_rate(epoch)

        for batch in batches:
            batch_cost, grad = f(x, batch)
            cost += batch_cost

            # Quick rmsprop implementation.
            # Divide the gradient by the square root of the running
            # average of the mean square of previous gradients.
            if rmsprop:
                for i, grad_i in enumerate(grad):
                    avg_sq_grad[i] = 0.9 * avg_sq_grad[i] + 0.1 * (grad_i ** 2)
                grad = tuple(grad_i / np.sqrt(avg_sq_grad[i])
                             for i, grad_i
                             in enumerate(grad))

            # Apply momentum and the learning rate.
            for i, grad_i in enumerate(grad):
                delta[i] = (p * delta[i]) + (e * grad_i)
                
            # delta = tuple((p * delta_i) + (e * grad_i)
            #               for delta_i, grad_i
            #               in zip(delta, grad))


            # TODO: Is imap necessary here, can't I just use a
            # generator expression?
            x = make_tuple(itertools.imap(operator.sub, x, delta))
            #x = make_tuple(x[i] - delta[i] for i in range(len(x)))

            if not weight_constraint is None:
                x = make_tuple(weight_constraint(x_i) for x_i in x)

            # import pdb; pdb.set_trace()

            # I've removed this as it was filling up the undo buffer
            #when I working interactively in Emacs.
            #sys.stdout.write('mini-batch %i\r' % (batch))
            #sys.stdout.flush()

        print 'epoch %3i, cost %0.6f' % (epoch, cost/len(batches))

        if post_epoch is not None:
            post_epoch(x, epoch, cost/len(batches))
    
    duration = datetime.timedelta(seconds=int(time.time() - t0))
    print '\nTraining took %s' % duration

    return x

def compute_gradient(f, x):
    # Note that x is expected to be a vector and not a tuple of params
    # as I use elsewhere.
    epsilon = 1e-6
    n = len(x)
    eye = np.identity(n)
    gradient = np.zeros(n)
    for i in range(n):
       delta = eye[i] * epsilon
       gradient[i] = (f(x + delta) - f(x - delta)) / (2 * epsilon)
    return gradient

# To re-pack params you need to hang on to shapes returned by pack,
# and you also usually need to take care of converting the result of
# pack_params back into the correct named tuple type. (See
# check_gradient for an example.) A better idea might be to have a
# single function which takes params as an arg as returns pack and
# unpack functions suitable for params tuples of the same size and
# type.
def unpack_params(params):
    theta = np.hstack(map(np.ravel, params))
    shapes = map(np.shape, params)
    return theta, shapes

def pack_params(theta, shapes):
    indexes = np.cumsum([0] + map(np.prod, shapes))
    params = []
    for i, shape in enumerate(shapes):
        start = indexes[i]
        end = indexes[i+1]
        p = theta[start:end].reshape(shape)
        params.append(p)
    return params

def _make_tuple_func(t):
    assert isinstance(t, tuple), 't expected to be a tuple'
    if hasattr(type(t), '_make'):
        make_tuple = type(t)._make # namedtuple
    else:
        make_tuple = tuple
    return make_tuple

def check_gradient(f, params):
    """
    Compute the norm of the difference between the exact gradient as
    computed by f and its numerical approximation, at params.
    
    params is expected to be a tuple of ndarrays.

    f is expected to be a function of one argument which returns a
    (cost, gradient) tuple.
    """
    # Compute the exact gradient and unpack in to a single vector.
    gradient, _ = unpack_params(f(params)[1])

    # Compute the numerical approximation of the gradient.
    theta, shapes = unpack_params(params)
    make_tuple = _make_tuple_func(params)

    def cost(x):
        # Re-pack the single vector in to a params tuple.
        params = make_tuple(pack_params(x, shapes))
        # Return the cost.
        return f(params)[0]

    approx_gradient = compute_gradient(cost, theta)

    return np.linalg.norm(gradient - approx_gradient)

def constrain_l2_norm(a, max_norm):
    """
    """
    # Pass through 1d arrays to handle the case where a represents
    # bias terms rather than weight vectors.
    if a.ndim == 1:
        return a
    # Compute the norm of each row.
    norm = np.sqrt(np.sum(a ** 2, 1))
    mask = (norm > max_norm) * 1.
    scaling_factor = max_norm / norm
    scaling_factor = scaling_factor * mask
    scaling_factor = scaling_factor + (1 - mask)
    return a * scaling_factor[:,np.newaxis]

def _wrap(val):
    """
    """
    if callable(val):
        return val
    else:
        def f(*args):
            return val
        return f
