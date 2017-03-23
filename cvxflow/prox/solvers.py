
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.contrib import linalg
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
    def __init__(self, prox_f, prox_g, A, B, c, num_columns=1, atol=1e-4,
                 rtol=1e-2, rho=1):
        self.prox_f = prox_f
        self.prox_g = prox_g
        self.A = A
        self.B = B
        self.c = ops.convert_to_tensor(c)
        self.atol = atol
        self.rtol = rtol
        self.rho = rho

        if len(self.c.get_shape()) == 0:
            self.c = array_ops.reshape(c, (1,1))
        elif len(self.c.get_shape()) == 1:
            self.c = array_ops.reshape(c, (int(self.c.get_shape()[0]), 1))

        p, n = A.shape
        m = B.shape[1]

        self.x = variables.Variable(array_ops.zeros(shape=(n,num_columns)))
        self.z = variables.Variable(array_ops.zeros(shape=(m,num_columns)))
        self.u = variables.Variable(array_ops.zeros(shape=(p,num_columns)))
        self.variables = [self.x, self.z, self.u]

    def iterate(self, (x, z, u)):
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

        n, p = A.shape[1], B.shape[1]
        eps_pri = (self.atol*math_ops.sqrt(math_ops.to_float(p)) +
                   self.rtol*math_ops.reduce_max(
                       [norm(Axp), norm(Bzp), norm(c)]))
        eps_dual = (self.atol*math_ops.sqrt(math_ops.to_float(n)) +
                    self.rtol*self.rho*norm(A.apply(u, adjoint=True)))

        return [xp, zp, up], [r_norm, s_norm, eps_pri, eps_dual]

    def solve(self, max_iters=1000, epoch_iters=10, verbose=True, sess=None):
        def cond(k, varz, residuals):
            return k < epoch_iters

        def body(k, varz, residuals):
            varzp, residualsp = self.iterate(varz)
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
            print("ADMM, rtol=%.2e, atol=%.2e" % (self.rtol, self.atol))
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
                print("Converged.")
            else:
                print("Max iterations reached.")
