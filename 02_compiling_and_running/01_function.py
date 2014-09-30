# Fill in the TODOs in this exercise, then run
# python 01_function.py to see if your solution works!
#
# This exercice ask you to compile a Theano functiont and call it to
# execute "x + y".
from theano import tensor as T
from theano import function


def evaluate(x, y, expr, x_value, y_value):
    F = function([x,y],expr)
    return F(x_value,y_value)


if __name__ == "__main__":
    x = T.iscalar()
    y = T.iscalar()
    z = x + y
    assert evaluate(x, y, z, 1, 2) == 3
    print "SUCCESS!"
