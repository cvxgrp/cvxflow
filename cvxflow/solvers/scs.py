"""Solve an LP using SCS on tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import time

import cvxpy as cvx
from numpy import *
import tensorflow as tf

from cvxflow.cones import proj_nonnegative, proj_cone
from cvxflow.solvers import conjugate_gradient
from cvxflow.equilibrate import equilibrate, balance
from cvxflow.cone_problem import TensorProblem
from cvxflow import vector_ops

PrimalVars = namedtuple("PrimalVars", ["x", "y", "tau"])
DualVars = namedtuple("DualVars", ["r", "s", "kappa"])
Cache = namedtuple("Cache", ["g_x", "g_y", "z_x_prev"])
Residuals = namedtuple(
    "Residuals", ["p_norm", "d_norm", "c_dot_x", "b_dot_y"])
Counters = namedtuple("Counters", ["total_cg_iters", "iters"])

class ScaledTensorProblem(object):
    """Rescale the problem so that b and c have unit norm."""

    def __init__(self, orig, equil_iters):
        self.orig = orig
        # Equilibrate.
        shape = (int(self.orig.b.get_shape()[0]),
                 int(self.orig.c.get_shape()[0]))
        d, e = balance(self.orig.A, self.orig.AT, shape, equil_iters)
        equil_sess = tf.Session()
        with equil_sess.as_default():
            self.d, self.e = equil_sess.run([d, e])
        self.sigma = 1. / tf.norm(orig.b * self.d)
        self.rho = 1. / tf.norm(orig.c * self.e)
        self.b = self.sigma * orig.b * self.d
        self.c = self.rho * orig.c * self.e

    def A(self, x):
        return tf.multiply(self.d, self.orig.A(tf.multiply(self.e, x)))

    def AT(self, y):
        return self.e*self.orig.AT(self.d*y)

    @property
    def cone_slices(self):
        return self.orig.cone_slices

def solve_scs_linear(problem, w_x, w_y, z_x_init, cg_tol):
    """Solve the SCS linear system using conjugate gradient.

    z_x = (I + A'A)^{-1}(w_x - A'w_y)
    z_y = w_y + Az_x
    """
    #cg = conjugate_gradient.ConjugateGradient(A, b, x_init)
    #state = cg.solve()

    def M(x):
        return x + problem.AT(problem.A(x))
    cg = conjugate_gradient.ConjugateGradient(M, w_x - problem.AT(w_y), z_x_init, tol=cg_tol)
    state = cg.solve()
    z_x = state.x
    cg_iters = state.k

    z_y = w_y + problem.A(z_x)
    return z_x, z_y, cg_iters

def init_cache(problem, cache):
    """Compute g = M^{-1}h."""
    n = int(problem.c.get_shape()[0])
    cg_tol = 1e-9
    g_x_init = tf.zeros((n, 1), dtype=tf.float32)
    g_x, g_y, _ = solve_scs_linear(
        problem, problem.c, problem.b, g_x_init, cg_tol)
    return tf.group(
        cache.g_x.assign(g_x),
        cache.g_y.assign(g_y))

def iterate(problem, u, v, cache, counters):
    """A single SCS iteration."""
    # u_tilde: solve linear system
    w_x = u.x + v.r
    w_y = u.y + v.s
    w_tau = u.tau + v.kappa

    cg_tol = 0.1 / tf.cast(tf.pow(counters.iters+1, 2), dtype=tf.float32)
    z_x, z_y, cg_iters = solve_scs_linear(
        problem, w_x, w_y, cache.z_x_prev, cg_tol)

    g_dot_w = tf.tensordot(cache.g_x, w_x, 2) + tf.tensordot(cache.g_y, w_y, 2)
    g_dot_h = tf.tensordot(cache.g_x, problem.c, 2) + tf.tensordot(cache.g_y, problem.b, 2)
    print (tf.tensordot(z_x, problem.c, 2))
    alpha = ((w_tau*g_dot_h -
              tf.tensordot(z_x, problem.c, 2) -
              tf.tensordot(z_y, problem.b, 2))/(1 + g_dot_h) - w_tau)

    u_tilde_x = z_x + alpha*cache.g_x
    u_tilde_y = z_y + alpha*cache.g_y
    u_tilde_tau = w_tau + tf.tensordot(u_tilde_x, problem.c, 2) + tf.tensordot(u_tilde_y,problem.b,  2)

    # u: cone projection
    u_x = u_tilde_x - v.r
    u_y = proj_cone(problem.cone_slices, u_tilde_y - v.s, dual=True)
    u_tau = proj_nonnegative(u_tilde_tau - v.kappa)

    # v: dual update
    v_r = v.r - u_tilde_x+ u_x
    v_s = v.s - u_tilde_y + u_y
    v_kappa = v.kappa - u_tilde_tau + u_tau

    return tf.group(
        u.x.assign(u_x),
        u.y.assign(u_y),
        u.tau.assign(u_tau),
        v.r.assign(v_r),
        v.s.assign(v_s),
        v.kappa.assign(v_kappa),
        cache.z_x_prev.assign(z_x),
        counters.total_cg_iters.assign(counters.total_cg_iters + cg_iters),
        counters.iters.assign(counters.iters + 1))

def compute_residuals(problem, u, v):
    """SCS residuals and duality gap."""
    x = u.x / u.tau
    y = u.y / u.tau
    s = v.s / u.tau
    p_norm = tf.norm(problem.A(x) + s - problem.b) / problem.sigma
    d_norm = tf.norm(problem.AT(y) + problem.c) / problem.rho
    c_dot_x = tf.tensordot(problem.c, x, 2) / problem.sigma / problem.rho
    b_dot_y = tf.tensordot(problem.b, y, 2) / problem.sigma / problem.rho
    return Residuals(p_norm, d_norm, c_dot_x, b_dot_y)

def create_variables(problem):
    m = int(problem.b.get_shape()[0])
    n = int(problem.c.get_shape()[0])

    u = PrimalVars(
        tf.Variable(tf.zeros((n,1), dtype=tf.float32)),
        tf.Variable(tf.zeros((m,1), dtype=tf.float32)),
        tf.Variable(tf.ones((1,1), dtype=tf.float32)))
    v = DualVars(
        tf.Variable(tf.zeros((n,1), dtype=tf.float32)),
        tf.Variable(tf.zeros((m,1), dtype=tf.float32)),
        tf.Variable(tf.ones((1,1), dtype=tf.float32)))

    return u, v

def create_cache(problem):
    m = int(problem.b.get_shape()[0])
    n = int(problem.c.get_shape()[0])

    return Cache(
        tf.Variable(tf.zeros((n,1), dtype=tf.float32)),
        tf.Variable(tf.zeros((m,1), dtype=tf.float32)),
        tf.Variable(tf.zeros((n,1), dtype=tf.float32)))

def create_counters():
    return Counters(
        tf.Variable(0, dtype=tf.int32),
        tf.Variable(0, dtype=tf.int32))


def solve(problem, max_iters=2500, trace=False, eps_primal=1e-3, eps_dual=1e-3,
          eps_gap=1e-3, equil_iters=50, gpu=True):
    """Create SCS tensorflow graph and solve."""
    scaled_problem = ScaledTensorProblem(problem, equil_iters)

    # variables
    u, v = create_variables(scaled_problem)
    cache = create_cache(scaled_problem)
    counters = create_counters()

    # ops
    t0 = time.time()
    init_op = tf.initialize_all_variables()
    init_cache_op = init_cache(scaled_problem, cache)
    iterate_op = iterate(scaled_problem, u, v, cache, counters)
    residuals = compute_residuals(scaled_problem, u, v)
    print("graph_build_time: %.2f secs" % (time.time() - t0))

    if trace:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    if gpu:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(device_count={"GPU": 0})

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        b_norm, c_norm = sess.run([tf.norm(problem.b), tf.norm(problem.c)])
        print("b_norm:", b_norm)
        print("c_norm:", c_norm)

        def check_converged_and_print(k):
            p_norm, d_norm, c_dot_x, b_dot_y, tau, kappa = sess.run([
                residuals.p_norm,
                residuals.d_norm,
                residuals.c_dot_x,
                residuals.b_dot_y,
                u.tau, v.kappa])

            g = c_dot_x + b_dot_y
            p_norm0 = p_norm / (1 + b_norm)
            d_norm0 = d_norm / (1 + c_norm)
            g0 = abs(g) / (1 + abs(c_dot_x) + abs(b_dot_y))

            if k % 100 == 0:
                print("k=%d, ||p||=%.4e, ||d||=%.4e, |g|=%.4e, pri=%.4e, dua=%.4e, kap/tau=%.4e" % (
                    k, p_norm0, d_norm0, g0, c_dot_x, -b_dot_y, kappa / tau))

            return (p_norm0 <= eps_primal and
                    d_norm0 <= eps_dual and
                    g0 <= eps_gap)

        sess.run(init_cache_op)
        for k in xrange(max_iters):
            if trace:
                sess.run(iterate_op, options=run_options,
                         run_metadata=run_metadata)


                tl = tf.python.client.timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open("scs_tf_iter_%d.json" % k, "w") as f:
                    f.write(ctf)
            else:
                sess.run(iterate_op)

            if k % 20 == 0 and check_converged_and_print(k):
                break
        else:
            check_converged_and_print(max_iters)

        iters = sess.run(counters.iters)
        total_cg_iters = sess.run(counters.total_cg_iters)
        objective = sess.run(residuals.c_dot_x)

    print("iterations:", k)
    print("avg_cg_iters: %.2f" % (float(total_cg_iters) / float(k)))
    return objective
