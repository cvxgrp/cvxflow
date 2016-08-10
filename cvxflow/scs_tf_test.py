"""Tests for SCS tensorflow."""

from numpy.testing import assert_allclose
import cvxpy as cvx
import numpy as np
import tensorflow as tf

from cvxflow import scs_tf
from cvxflow.problem import TensorProblem
from cvxflow.tf_util import vstack

def linear_program():
    np.random.seed(0)
    m = 5
    n = 10
    A = np.abs(np.random.randn(m,n))
    b = A.dot(np.abs(np.random.randn(n)))
    c = np.random.rand(n) + 0.5
    x = cvx.Variable(n)
    return cvx.Problem(cvx.Minimize(c.T*x), [A*x == b, x >= 0])

def expected_subspace_projection(data, x):
    A = data["A"].todense()
    b = data["b"].reshape(-1,1)
    c = data["c"].reshape(-1,1)

    m, n = data["A"].shape
    Q = np.zeros((m+n+1, m+n+1))
    Q[0:n, n:n+m]       =  A.T
    Q[0:n, n+m:n+m+1]   =  c
    Q[n:n+m, 0:n]       = -A
    Q[n:n+m, n+m:n+m+1] =  b
    Q[n+m:n+m+1, 0:n]   = -c.T
    Q[n+m:n+m+1, n:n+m] = -b.T
    return np.linalg.solve(np.eye(m+n+1) + Q, x)

def expected_cone_projection(data, x):
    dims = data["dims"]
    n = data["A"].shape[1]

    idx = slice(n+dims["f"], n+dims["f"]+dims["l"])
    x[idx] = np.maximum(x[idx], 0)
    x[-1] = np.maximum(x[-1], 0)
    return x

def test_scs():
    cvx_problem = linear_program()
    problem = TensorProblem(cvx_problem)
    data = cvx_problem.get_problem_data(cvx.SCS)

    # Compare with manually implemented SCS iterations
    m, n = data["A"].shape
    u0 = np.zeros((m+n+1,1))
    v0 = np.zeros((m+n+1,1))
    u0[-1] = 1
    v0[-1] = 1

    # Run two iterations
    u, v, cache = scs_tf.variables(problem)
    u_vec = vstack([u.x, u.y, u.tau])
    v_vec = vstack([v.r, v.s, v.kappa])

    init_op = tf.initialize_all_variables()
    init_scs_op = scs_tf.init_scs(problem, cache)
    iteration_op = scs_tf.iteration(problem, cache, u, v)
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(init_scs_op)

        print "initialization"
        assert_allclose(u0, sess.run(u_vec))
        assert_allclose(v0, sess.run(v_vec))

        # print "first iteration"
        u_tilde0 = expected_subspace_projection(data, u0 + v0)
        u0 = expected_cone_projection(data, u_tilde0 - v0)
        v0 = v0 - u_tilde0 + u0
        sess.run(iteration_op)
        assert_allclose(u0, sess.run(u_vec))
        assert_allclose(v0, sess.run(v_vec))

        # print "second iteration"
        u_tilde0 = expected_subspace_projection(data, u0 + v0)
        u0 = expected_cone_projection(data, u_tilde0 - v0)
        v0 = v0 - u_tilde0 + u0
        sess.run(iteration_op)
        assert_allclose(u0, sess.run(u_vec))
        assert_allclose(v0, sess.run(v_vec))
