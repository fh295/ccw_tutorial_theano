# Fill in the TODOs in this exercise, then run
# python 02_vector_mat.py to see if your solution works!
#
# This exercices ask you to make Theano variable, elemwise
# multiplication and matrix/vector dot product.
import numpy as np
from theano import function
import theano.tensor as T


def make_vector():
    return T.vector()


def make_matrix():
    return T.matrix()


def elemwise_mul(a, b):
    return a*b

def matrix_vector_mul(a, b):
    return T.dot(a,b)


if __name__ == "__main__":
    a = make_vector()
    b = make_vector()
    c = elemwise_mul(a, b)
    d = make_matrix()
    e = matrix_vector_mul(d, c)

    f = function([a, b, d], e)

    rng = np.random.RandomState([1, 2, 3])
    a_value = rng.randn(5).astype(a.dtype)
    b_value = rng.rand(5).astype(b.dtype)
    c_value = a_value * b_value
    d_value = rng.randn(5, 5).astype(d.dtype)
    expected = np.dot(d_value, c_value)

    actual = f(a_value, b_value, d_value)

    assert np.allclose(actual, expected)
    print "SUCCESS!"
