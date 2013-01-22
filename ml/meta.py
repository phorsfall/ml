# Helper functions for meta parameters.

# I intentionally don't set defaults here to force myself to specify
# them explicitly in calling code. This is because a) I don't know
# that any values I may pick will work well across problems and b)
# having the values in the code helps ensure reproducability.

def step(initial, final, T):
    """
    """
    def f(t):
        if t < T:
            return initial
        else:
            return final
    return f

def linear(initial, final, T):
    """
    Increase linearly from initial to final over T then remain at
    final.
    """
    def f(t):
        if t < T:
            r = t / float(T)
            p = (1 - r) * initial + r * final
        else:
            p = final
        return p
    return f

def exponential(initial, decay_rate):
    """
    """
    def f(t):
        return initial * decay_rate ** t
    return f
