"""Tests for SCS tensorflow."""

from numpy.testing import assert_allclose
import cvxpy as cvx
import numpy as np
import tensorflow as tf

from cvxflow import scs_tf
from cvxflow.problem import TensorProblem
from cvxflow.problem_testutil import PROBLEMS
from cvxflow.tf_util import vstack

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

def proj_second_order(x):
    return x

def expected_cone_projection(data, x):
    dims = data["dims"]
    n = data["A"].shape[1]
    offset = n + dims["f"]

    idx = slice(offset, offset+dims["l"])
    x[idx] = np.maximum(x[idx], 0)
    offset += dims["l"]

    for qi in dims["q"]:
        idx = slice(offset, offset+qi)
        x[idx] = proj_second_order(x[idx])
        offset += qi

    x[-1] = np.maximum(x[-1], 0)
    return x

def test_problems():
    for problem in PROBLEMS:
        yield run_problem, problem()

def run_problem(cvx_problem):
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

        print "first iteration"
        u_tilde0 = expected_subspace_projection(data, u0 + v0)
        u0 = expected_cone_projection(data, u_tilde0 - v0)
        v0 = v0 - u_tilde0 + u0
        sess.run(iteration_op)
        assert_allclose(u0, sess.run(u_vec), rtol=0, atol=1e-6)
        assert_allclose(v0, sess.run(v_vec), rtol=0, atol=1e-6)

        print "second iteration"
        u_tilde0 = expected_subspace_projection(data, u0 + v0)
        u0 = expected_cone_projection(data, u_tilde0 - v0)
        v0 = v0 - u_tilde0 + u0
        sess.run(iteration_op)
        assert_allclose(u0, sess.run(u_vec), rtol=0, atol=1e-6)
        assert_allclose(v0, sess.run(v_vec), rtol=0, atol=1e-6)
