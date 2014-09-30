# Fill in the TODOs in this exercise, then run
# python 01_grad.py to see if your solution works!
#
# This exercice ask you to use Theano automatic gradient system to
# compute some derivative.
from theano import tensor as T


def grad_sum(x, y, z):
    return sum(T.grad(z,x), T.grad(z,y))



if __name__ == "__main__":
    x = T.scalar()
    y = T.scalar()
    z = x + y
    s = grad_sum(x, y, z)
    assert s.eval({x: 0, y:0}) == 2
    print "SUCCESS!"
