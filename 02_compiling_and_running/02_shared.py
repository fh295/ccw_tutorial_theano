# Fill in the TODOs in this exercise, then run
# python 01_function.py to see if your solution works!
#
# This exercice make you use shared variable. You must create them and
# update them by swapping 2 shared variables values.
import numpy as np
import theano
from theano import tensor as T
from theano import shared, function

def make_shared(shape):
    return shared(np.zeros(shape))
    
def exchange_shared(a, b):
    a.set_value = b.get_value
    b.set_value = a.get_value
    

def make_exchange_func(a, b):
    F = exchange_shared(a,b)
    f = function([a,b],F)
    return f
    




if __name__ == "__main__":
    a = make_shared((5, 4, 3))
    assert a.get_value().shape == (5, 4, 3)
    b = make_shared((5, 4, 3))
    assert a.get_value().shape == (5, 4, 3)
    a.set_value(np.zeros((5, 4, 3), dtype=a.dtype))
    b.set_value(np.ones((5, 4, 3), dtype=b.dtype))
    exchange_shared(a, b)
    assert np.all(a.get_value() == 1.)
    assert np.all(b.get_value() == 0.)
    f = make_exchange_func(a, b)
    rval = f()
    assert isinstance(rval, list)
    assert len(rval) == 0
    assert np.all(a.get_value() == 0.)
    assert np.all(b.get_value() == 1.)

    print "SUCCESS!"
