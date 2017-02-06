import theano
from theano import tensor

a = tensor.dscalar()
b = tensor.dscalar()
f = theano.function([a, b], a + b)
result = f(1.5, 2.5)
print result