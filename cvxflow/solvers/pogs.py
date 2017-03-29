from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np
import tensorflow as tf
import time

from cvxflow.solvers import conjugate_gradient
from cvxflow.solvers.iterative_solver import IterativeSolver, run_epochs

PROJECT_TOL_MIN = 1e-2
PROJECT_TOL_MAX = 1e-2
PROJECT_TOL_POW = 1.3

class POGS(IterativeSolver):
    """Proximal Operator Graph Solver.

    minimize    f(y) + g(x)
    subject to  y = Ax
    """
    def __init__(
            self, prox_f=None, prox_g=None, A=None, AT=None, shape=None,
            dtype=tf.float32, rho=1, rtol=1e-2, atol=1e-4, max_iterations=10000,
            **kwargs):
        self.prox_f = prox_f or (lambda x: x)
        self.prox_g = prox_g or (lambda x: x)
        self.A = A
        self.AT = AT
        self.m, self.n = shape
        self.dtype = dtype
        self.atol = atol
        self.rtol = rtol
        self.rho = rho
        self.max_iterations = max_iterations
        super(POGS, self).__init__(**kwargs)

    @property
    def state(self):
        return namedtuple(
            "State", [
                "x", "y",
                "x_tilde", "y_tilde",
                "r_norm", "s_norm",
                "eps_pri", "eps_dual",
                "k", "total_cg_iters"])

    def init(self):
        x = tf.constant(0, shape=(self.n,1), dtype=self.dtype)
        y = tf.constant(0, shape=(self.m,1), dtype=self.dtype)
        x_tilde = tf.constant(0, shape=(self.n,1), dtype=self.dtype)
        y_tilde = tf.constant(0, shape=(self.m,1), dtype=self.dtype)
        r_norm = tf.constant(np.inf, dtype=self.dtype)
        s_norm = tf.constant(np.inf, dtype=self.dtype)
        eps_pri = tf.constant(0, dtype=self.dtype)
        eps_dual = tf.constant(0, dtype=self.dtype)
        k = tf.constant(0)
        total_cg_iters = tf.constant(0)
        return self.state(
            x, y,
            x_tilde, y_tilde,
            r_norm, s_norm,
            eps_pri, eps_dual,
            k, total_cg_iters)

    def iterate(self, state):
        with tf.name_scope("prox"):
            x_h = self.prox_g(state.x - state.x_tilde)
            y_h = self.prox_f(state.y - state.y_tilde)

        with tf.name_scope("projection"):
            k = tf.cast(state.k, self.dtype)
            cg_tol = tf.maximum(
                PROJECT_TOL_MIN / tf.pow(k+1, PROJECT_TOL_POW),
                PROJECT_TOL_MAX)

            x0 = x_h + state.x_tilde
            y0 = y_h + state.y_tilde
            b = y0 - self.A(x0)
            x_init = tf.zeros_like(x0)
            cgls = conjugate_gradient.ConjugateGradientLeastSquares(
                self.A, self.AT, b, x_init, tol=cg_tol, shift=1)
            cg_state = cgls.solve()
            x = cg_state.x + x0
            y = self.A(x)
            total_cg_iters =  state.total_cg_iters + cg_state.k

        with tf.name_scope("dual_update"):
            x_tilde = x0 - x
            y_tilde = y0 - y

        # TODO(mwytock): Only run this every k iterations
        with tf.name_scope("residuals"):
            mu_h = -self.rho*(x_h - state.x + state.x_tilde)
            nu_h = -self.rho*(y_h - state.y + state.y_tilde)
            eps_pri = self.atol + self.rtol*tf.norm(y_h)
            eps_dual = self.atol + self.rtol*tf.norm(mu_h)
            r_norm = tf.norm(self.A(x_h) - y_h)
            s_norm = tf.norm(self.AT(nu_h) + mu_h)

        return self.state(
            x, y,
            x_tilde, y_tilde,
            r_norm, s_norm,
            eps_pri, eps_dual,
            state.k+1, total_cg_iters)

    def stop(self, state):
        return tf.logical_or(
            tf.logical_and(state.r_norm <= state.eps_pri,
                           state.s_norm <= state.eps_dual),
            state.k >= self.max_iterations)


def run(sess, epoch_iterations=10, **kwargs):
    pogs = POGS(**kwargs)
    print("POGS - proximal operator graph solver")
    print("rtol=%.2e, atol=%.2e" % (pogs.rtol, pogs.atol))
    print("%5s %10s %10s %10s %10s %6s" % (
        ("iter", "r norm", "eps pri", "s norm", "eps dual", "time")))
    print("-"*56)

    t0 = time.time()
    def status(state):
        print("%5d %10.2e %10.2e %10.2e %10.2e %5.0fs" %
              sess.run(state.k, state.r_norm, state.eps_pri, state.s_norm,
                       state.eps_dual) + [time.time() - t0])

    state = run_epochs(sess, pogs, epoch_iterations, status)
    status = "Converged" if state.k <= pogs.max_iterations else "Max iterations"
    print("-"*56)
    print("%s, %.2f seconds" % (status, time.time() - t0))
    print("Average CGLS iters: %.2f" % (state.total_cg_iters / state.k))
