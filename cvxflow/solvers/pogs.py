from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np
import tensorflow as tf
import time

from cvxflow.solvers import conjugate_gradient
from cvxflow.solvers.iterative_solver import IterativeSolver, run_epochs
from cvxflow import utils

PROJECT_TOL_MIN = 1e-2
PROJECT_TOL_MAX = 1e-8
PROJECT_TOL_POW = 1.3

class POGS(IterativeSolver):
    """Proximal Operator Graph Solver.

    minimize    f(y) + g(x)
    subject to  y = Ax
    """
    def __init__(
            self, prox_f=None, prox_g=None, A=None, AT=None, shape=None,
            dtype=tf.float32, rho=1, rtol=1e-2, atol=1e-4, max_iterations=10000,
            residual_iterations=100, project_linear=None, **kwargs):
        self.prox_f = prox_f or (lambda x: x)
        self.prox_g = prox_g or (lambda x: x)
        self.A = A
        self.AT = AT or A
        self.x_shape, self.y_shape = shape
        self.dtype = dtype
        self.atol = atol
        self.rtol = rtol
        self.rho = rho
        self.max_iterations = max_iterations
        self.residual_iterations = residual_iterations
        self.project_linear = project_linear

        self.state = namedtuple("State", [
            "x", "y",
            "x_tilde", "y_tilde",
            "r_norm", "s_norm",
            "eps_pri", "eps_dual",
            "k", "total_cg_iters"])

        super(POGS, self).__init__(**kwargs)

    def init(self):
        x = tf.constant(0, shape=self.x_shape, dtype=self.dtype)
        y = tf.constant(0, shape=self.y_shape, dtype=self.dtype)
        x_tilde = tf.constant(0, shape=self.x_shape, dtype=self.dtype)
        y_tilde = tf.constant(0, shape=self.y_shape, dtype=self.dtype)
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

    def project_cgls(self, x0, y0, tol):
        b = y0 - self.A(x0)
        x_init = tf.zeros_like(x0)
        cgls = conjugate_gradient.ConjugateGradientLeastSquares(
            self.A, self.AT, b, x_init, tol=tol, shift=1)
        cg_state = cgls.solve()
        x = cg_state.x + x0
        y = self.A(x)
        return x, y, cg_state.k

    def iterate(self, state):
        with tf.name_scope("prox"):
            x_h = self.prox_g(state.x - state.x_tilde)
            y_h = self.prox_f(state.y - state.y_tilde)

        with tf.name_scope("projection"):
            x0 = x_h + state.x_tilde
            y0 = y_h + state.y_tilde

            if self.project_linear:
                x, y = self.project_linear(x0, y0)
                cg_iters = 0
            else:
                k = tf.cast(state.k, self.dtype)
                cg_tol = tf.maximum(
                    PROJECT_TOL_MIN / tf.pow(k+1, PROJECT_TOL_POW),
                    PROJECT_TOL_MAX)
                x, y, cg_iters = self.project_cgls(x0, y0, cg_tol)

        with tf.name_scope("dual_update"):
            x_tilde = x0 - x
            y_tilde = y0 - y

        with tf.name_scope("residuals"):
            # Only calculate residuals every k iterations
            def calculate_residuals():
                n = np.prod(self.x_shape)
                m = np.prod(self.y_shape)
                mu_h = -self.rho*(x_h - state.x + state.x_tilde)
                nu_h = -self.rho*(y_h - state.y + state.y_tilde)
                eps_pri = self.atol*np.sqrt(m) + self.rtol*tf.norm(y_h)
                eps_dual = self.atol*np.sqrt(n) + self.rtol*tf.norm(mu_h)
                r_norm = tf.norm(self.A(x_h) - y_h)
                s_norm = tf.norm(self.AT(nu_h) + mu_h)
                return r_norm, s_norm, eps_pri, eps_dual

            r_norm, s_norm, eps_pri, eps_dual = (
                tf.cond(tf.equal((state.k+1) % self.residual_iterations, 0),
                        calculate_residuals,
                        lambda: (state.r_norm, state.s_norm,
                                 state.eps_pri, state.eps_dual)))

        return self.state(
            x, y,
            x_tilde, y_tilde,
            r_norm, s_norm,
            eps_pri, eps_dual,
            state.k+1,
            state.total_cg_iters+cg_iters)

    def stop(self, state):
        return tf.logical_or(
            tf.logical_and(state.r_norm <= state.eps_pri,
                           state.s_norm <= state.eps_dual),
            state.k >= self.max_iterations)


def run(sess, epoch_iterations=10, profile=False, **kwargs):
    pogs = POGS(**kwargs)
    print("POGS - proximal operator graph solver")
    print("m=%d, n=%d, rtol=%.2e, atol=%.2e" % (
        pogs.m, pogs.n, pogs.rtol, pogs.atol))

    print("%5s %10s %10s %10s %10s %10s %6s" % (
        ("iter", "r norm", "eps pri", "s norm", "eps dual", "cg iters", "time")))
    print("-"*67)

    t0 = time.time()
    last_values = [t0, 0, 0]
    def status(state):
        t0, last_total_cg_iters, last_k = last_values
        values = sess.run([state.k, state.r_norm, state.eps_pri,
                           state.s_norm, state.eps_dual])

        k, total_cg_iters = sess.run([state.k, state.total_cg_iters])
        avg_cg_iters = (total_cg_iters - last_total_cg_iters) / (k - last_k)
        values.append(avg_cg_iters)

        t1 = time.time()
        values.append(t1 - t0)

        print("%5d %10.2e %10.2e %10.2e %10.2e %10.2f %5.0fs" % tuple(values))
        last_values[:] = t1, total_cg_iters, k

    state = run_epochs(sess, pogs, epoch_iterations, status, profile=profile)
    total_cg_iters, total_iters = sess.run([state.total_cg_iters, state.k])
    if total_iters >= pogs.max_iterations:
        status = "Max iterations reached"
    else:
        status = "Converged"

    print("-"*67)
    print("%s, %.2f seconds" % (status, time.time() - t0))
