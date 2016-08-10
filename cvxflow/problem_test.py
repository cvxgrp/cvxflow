
from nose.tools import assert_items_equal
import cvxpy as cvx
import numpy as np
import tensorflow as tf

from cvxflow import tf_problem

np.random.seed(0)
m = 5
n = 10
A = np.abs(np.random.randn(m,n))
b = A.dot(np.abs(np.random.randn(n)))
c = np.random.rand(n)
x = cvx.Variable(n)

def assert_tensor(sess, tensor, expected, tol=1e-8):
    actual = sess.run(tensor)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tol)

def test_tensor_problem():
    cvx_problem = cvx.Problem(cvx.Minimize(c.T*x), [A*x == b, x >= 0])
    data = cvx_problem.get_problem_data(cvx.SCS)
    m, n = data["A"].shape
    x0 = np.random.randn(n,1)
    y0 = np.random.randn(m,1)
    x = tf.constant(x0)
    y = tf.constant(y0)

    problem = tf_problem.TensorProblem(cvx_problem)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        assert_tensor(sess, problem.b, data["b"])
        assert_tensor(sess, problem.c, data["c"])
        assert_tensor(sess, problem.A(x), data["A"]*x0)
        assert_tensor(sess, problem.AT(y), data["A"].T*y0)
