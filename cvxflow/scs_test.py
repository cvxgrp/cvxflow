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
    s, v = x[:1,:], x[1:, :]
    norm_v = np.linalg.norm(v)

    if norm_v <= -s:
        return np.zeros(x.shape)
    elif norm_v <= s:
        return x
    else:
        return 0.5*(1 + s/norm_v)*np.vstack((norm_v, v))

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

def test_iterates():
    for problem in PROBLEMS:
        yield run_iterates, problem

class TestTensorProblem(object):
    def __init__(self, orig):
        self.orig = orig
        self.sigma = 1.
        self.rho = 1.
        self.b = self.sigma * orig.b
        self.c = self.rho * orig.c

    def A(self, x):
        return self.orig.A(x)

    def AT(self, y):
        return self.orig.AT(y)

    @property
    def cone_slices(self):
        return self.orig.cone_slices

class IterateTest(tf.test.TestCase):
    pass

def get_iterate_test(problem_gen):
    def test(self):
        np.random.seed(0)
        cvx_problem = problem_gen()

        problem = TestTensorProblem(TensorProblem(cvx_problem))
        data = cvx_problem.get_problem_data(cvx.SCS)

        # Compare with manually implemented SCS iterations
        m, n = data["A"].shape
        u0 = np.zeros((m+n+1,1))
        v0 = np.zeros((m+n+1,1))
        u0[-1] = 1
        v0[-1] = 1

        # variables
        u, v = scs_tf.create_variables(problem)
        cache = scs_tf.create_cache(problem)
        counters = scs_tf.create_counters()

        # ops
        init_op = tf.initialize_all_variables()
        init_cache_op = scs_tf.init_cache(problem, cache)
        iterate_op = scs_tf.iterate(problem, u, v, cache, counters)
        residuals = scs_tf.compute_residuals(problem, u, v)

        # Run two iterations
        u_vec = vstack([u.x, u.y, u.tau])
        v_vec = vstack([v.r, v.s, v.kappa])

        with self.test_session():
            init_op.run()
            init_cache_op.run()

            tf.logging.info("initialization")
            assert_allclose(u0, u_vec.eval())
            assert_allclose(v0, v_vec.eval())

            tf.logging.info("first iteration")
            iterate_op.run()
            u_tilde0 = expected_subspace_projection(data, u0 + v0)
            u0 = expected_cone_projection(data, u_tilde0 - v0)
            v0 = v0 - u_tilde0 + u0
            assert_allclose(u0, u_vec.eval(), rtol=0, atol=1e-4)
            assert_allclose(v0, v_vec.eval(), rtol=0, atol=1e-4)

            u0 = u_vec.eval()
            v0 = v_vec.eval()

            tf.logging.info("second iteration")
            iterate_op.run()
            u_tilde0 = expected_subspace_projection(data, u0 + v0)
            u0 = expected_cone_projection(data, u_tilde0 - v0)
            v0 = v0 - u_tilde0 + u0
            assert_allclose(u0, u_vec.eval(), rtol=0, atol=1e-2)
            assert_allclose(v0, v_vec.eval(), rtol=0, atol=1e-2)

    return test

if __name__ == "__main__":
    for problem in PROBLEMS:
        test_name = "test_%s" % problem.__name__
        setattr(IterateTest, test_name, get_iterate_test(problem))
    tf.test.main()
