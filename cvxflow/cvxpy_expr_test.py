
from numpy.testing import assert_allclose
import cvxpy as cvx
import tensorflow as tf
import numpy as np

from cvxflow import cvxpy_expr

x = cvx.Variable(3)
EXPRESSIONS = [
    (cvx.conv([1,2,3], x), [1,2,4])
]

def run_tensor(expr, x0):
    x_t = tf.constant(x0.reshape(-1,1), dtype=tf.float32)
    expr_t = cvxpy_expr.tensor(expr.canonicalize()[0], {x.id: x_t})
    x.value = x0
    with tf.Session() as sess:
        print sess.run(expr_t)
        print expr.value
        assert_allclose(sess.run(expr_t), expr.value)

def test_tensor_conv():
    for f, x0 in EXPRESSIONS:
        yield run_tensor, f, np.array(x0)
