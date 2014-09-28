# Fill in the TODOs in this exercise, then run
# python 03_tensor.py to see if your solution works!
#
# This exercices ask you to create Theano tensor variable, do
# broadcastable addition and to compute the max over part of a tensor.
import numpy as np
from theano import function
import theano.tensor as T

def make_tensor(dim):
    return T.tensor('float64', (False)*dim)()

def broadcasted_add(a, b):
    a1 = a.dimshuffle((2,'x',1,0))
    return a1 + b

def partial_max(a):
    return a.max(axis=(1,2), keepdims=False)


if __name__ == "__main__":
    a = make_tensor(3)
    b = make_tensor(4)
    c = broadcasted_add(a, b)
    d = partial_max(c)

    f = function([a, b,], d)

    rng = np.random.RandomState([1, 2, 3])
    a_value = rng.randn(2, 2, 2).astype(a.dtype)
    b_value = rng.rand(2, 2, 2, 2).astype(b.dtype)
    c_value = np.transpose(a_value, (2, 1, 0))[:, None, :, :] + b_value
    expected = c_value.max(axis=1).max(axis=1)

    actual = f(a_value, b_value)

    assert np.allclose(actual, expected), (actual, expected)
    print "SUCCESS!"
