
import cvxpy as cvx
import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from cvxflow.problem import TensorProblem
from cvxflow.problem_testutil import PROBLEMS

def run_problem(problem):
    cvx_problem = problem()
    data = cvx_problem.get_problem_data(cvx.SCS)
    m, n = data["A"].shape
    x0 = np.random.randn(n,1)
    y0 = np.random.randn(m,1)
    x = tf.constant(x0, dtype=tf.float32)
    y = tf.constant(y0, dtype=tf.float32)

    problem = TensorProblem(cvx_problem)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        assert_allclose(sess.run(problem.b), data["b"].reshape(-1,1), rtol=0, atol=1e-6)
        assert_allclose(sess.run(problem.c), data["c"].reshape(-1,1), rtol=0, atol=1e-6)
        assert_allclose(sess.run(problem.A(x)), data["A"]*x0, rtol=0, atol=1e-6)
        assert_allclose(sess.run(problem.AT(y)), data["A"].T*y0, rtol=0, atol=1e-6)

def test_problems():
    for problem in PROBLEMS:
        yield run_problem, problem
