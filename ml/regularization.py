import numpy as np

# Weight decay.

# I'd expect every function in here to have the logic required to skip
# bias terms. This could probably be extracted out into a decorator
# that wraps the core functionality of computing the cost and grad for
# any narray.

# It's tempting to see regularization as something which just wraps
# any cost function. We'd just that the cost/grad returned by that
# function and apply weight decay appropriately. This way, we could
# add weight decay to any cost function with a decorator (if we could
# parameterize the kind of decay easily) or by manually wrapping:
# cost_func = reg.decay_weights(cost_func, reg.l2(0.05))
# This would save having to implement it in each cost func. The only
# reason I've not done this yet is that a) I'm currently only using
# this module with RBMs and b) I'm not sure there won't be cases where
# something more subtle needs to be done that couldn't be achieved
# with this approach. If I use this a few times and it still seems
# like a good idea I might do it.

def l1(penalty, decay_biases=False):
    def f(a):
        if a.ndim == 2 or decay_biases:
            cost = penalty * np.sum(np.abs(a))
            grad = penalty * np.sign(a)
            return cost, grad
        else:
            cost = grad = 0.
            return cost, grad
    return f

def l2(penalty, decay_biases=False):
    def f(a):
        if a.ndim == 2 or decay_biases:
            v = a.ravel()
            cost = penalty * 0.5 * v.dot(v)
            grad = penalty * a
            return cost, grad
        else:
            cost = grad = 0.
            return cost, grad
    return f

# Weight constraints.

def l2_constraint(l2_max):
    def f(a):
        # Pass through 1d arrays to handle the case where a represents
        # bias terms rather than weight vectors.
        if a.ndim == 1:
            return a
        # Compute the l2 norm of each row.
        norm = np.sqrt(np.sum(a ** 2, 1))
        mask = (norm > l2_max) * 1.
        scaling_factor = l2_max / norm
        scaling_factor = scaling_factor * mask
        scaling_factor = scaling_factor + (1 - mask)
        return a * scaling_factor[:,np.newaxis]
    return f
