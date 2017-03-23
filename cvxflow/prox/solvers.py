
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

def norm(x):
    """L2-norm, elementwise."""
    return math_ops.sqrt(math_ops.reduce_sum(math_ops.square(x)))

class Solver(object):
    pass

class ADMM(Solver):
    """Alternating direction method of multipliers.

    minimize    f(x) + g(z)
    subject to  Ax - Bz = c
    """
    def __init__(self, prox_f, prox_g, A, B, c=0., num_columns=1, rho=1):
        self.prox_f = prox_f
        self.prox_g = prox_g
        self.A = A
        self.B = B
        self.rho = rho

        if A.dtype != B.dtype:
            raise ValueError("A and B must have same dtype.")
        self.dtype = A.dtype
        self.c = ops.convert_to_tensor(c, dtype=self.dtype)

        if len(self.c.get_shape()) == 0:
            self.c = array_ops.reshape(c, (1,1))
        elif len(self.c.get_shape()) == 1:
            self.c = array_ops.reshape(c, (int(self.c.get_shape()[0]), 1))

        p, n = A.shape
        m = B.shape[1]
        self.x = variables.Variable(
            array_ops.zeros(shape=(n,num_columns), dtype=self.dtype))
        self.z = variables.Variable(
            array_ops.zeros(shape=(m,num_columns), dtype=self.dtype))
        self.u = variables.Variable(
            array_ops.zeros(shape=(p,num_columns), dtype=self.dtype))
        self.variables = [self.x, self.z, self.u]

        self.n, self.m, self.p = [int(x*num_columns) for x in [n,m,p]]

    def iterate(self, (x, z, u), (rtol, atol)):
        A, B, c = self.A, self.B, self.c

        Bz = B.apply(z)
        xp = self.prox_f(A.apply(Bz - u + c, adjoint=True))

        Axp = A.apply(xp)
        zp = self.prox_g(B.apply(Axp + u - c, adjoint=True))

        Bzp = B.apply(zp)
        r = Axp - Bzp - c
        up = u + r

        r_norm = norm(r)
        s_norm = self.rho*norm(A.apply(Bzp - Bz, adjoint=True))

        eps_pri = (atol*np.sqrt(self.p) +
                   rtol*math_ops.reduce_max(
                       [norm(Axp), norm(Bzp), norm(c)]))
        eps_dual = (atol*np.sqrt(self.n) +
                    rtol*self.rho*norm(A.apply(u, adjoint=True)))

        return [xp, zp, up], [r_norm, s_norm, eps_pri, eps_dual]

    def solve(self, max_iters=10000, epoch_iters=10, verbose=True, sess=None,
              atol=1e-4, rtol=1e-2):
        t_start = time.time()
        tol = (rtol, atol)

        def cond(k, varz, residuals):
            return k < epoch_iters

        def body(k, varz, residuals):
            varzp, residualsp = self.iterate(varz, tol)
            return [k+1, varzp, residualsp]

        loop_vars = [0, self.variables, [0.,0.,0.,0.]]
        _, varz_epoch, residuals_epoch = control_flow_ops.while_loop(
            cond, body, loop_vars)
        init_op = variables.global_variables_initializer()
        epoch_op = control_flow_ops.group(
            *[x.assign(val) for x, val in zip(self.variables, varz_epoch)])

        if sess is None:
            sess = session.Session()

        if verbose:
            print("Starting ADMM...")
            print("n=%d, m=%d, p=%d" % (self.n, self.m, self.p))
            print("rtol=%.2e, atol=%.2e" % (rtol, atol))
            print("%5s %10s %10s %10s %10s %6s" % (
                ("iter", "r norm", "eps pri", "s norm", "eps dual", "time")))
            print("-"*56)

        sess.run(init_op)
        num_epochs = max_iters // epoch_iters
        for i in range(num_epochs):
            t0 = time.time()
            sess.run(epoch_op)
            r_norm0, s_norm0, eps_pri0, eps_dual0 = sess.run(residuals_epoch)
            t1 = time.time()

            if verbose:
                total_iters = (i+1)*epoch_iters
                print("%5d %10.2e %10.2e %10.2e %10.2e %5.0fs" %
                      (total_iters, r_norm0, eps_pri0, s_norm0, eps_dual0, t1 - t0))

            if r_norm0 < eps_pri0 and s_norm0 < eps_dual0:
                break

        if verbose:
            print("-"*56)
            if i < num_epochs - 1:
                status = "Converged"
            else:
                status = "Max iterations"

            print("%s, %.2f seconds." % (status, time.time() - t_start))

        return sess.run(self.variables)
