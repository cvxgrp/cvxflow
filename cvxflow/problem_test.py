
import cvxpy as cvx
import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from cvxflow.problem import TensorProblem

def linear_program():
    np.random.seed(0)
    m = 5
    n = 10
    A = np.abs(np.random.randn(m,n))
    b = A.dot(np.abs(np.random.randn(n)))
    c = np.random.rand(n)
    x = cvx.Variable(n)
    return cvx.Problem(cvx.Minimize(c.T*x), [A*x == b, x >= 0])

def test_linear():
    cvx_problem = linear_program()
    data = cvx_problem.get_problem_data(cvx.SCS)
    m, n = data["A"].shape
    x0 = np.random.randn(n,1)
    y0 = np.random.randn(m,1)
    x = tf.constant(x0)
    y = tf.constant(y0)

    problem = TensorProblem(cvx_problem)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        assert_allclose(sess.run(problem.b), data["b"].reshape(-1,1))
        assert_allclose(sess.run(problem.c), data["c"].reshape(-1,1))
        assert_allclose(sess.run(problem.A(x)), data["A"]*x0)
        assert_allclose(sess.run(problem.AT(y)), data["A"].T*y0)
