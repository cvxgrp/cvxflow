
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

class Solver(object):
    pass

class ADMM(Solver):
    """Alternating direction method of multipliers.

    minimize    f(x) + g(z)
    subject to  Ax - Bz = c
    """
    def __init__(self, prox_f, prox_g, A, B, c=None, num_columns=1, rho=1):
        self.prox_f = prox_f
        self.prox_g = prox_g
        self.A = A
        self.B = B
        self.rho = rho

        if A.dtype != B.dtype:
            raise ValueError("A and B must have same dtype.")
        self.dtype = A.dtype
        self.c = constant_op.constant(0, dtype=self.dtype) if c is None else c

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

    def zero_residuals(self):
        return [constant_op.constant(0, dtype=self.dtype) for _ in range(4)]

    def iterate(self, (x, z, u), (rtol, atol), need_residuals):
        A, B, c = self.A, self.B, self.c

        with ops.name_scope("x_update"):
            Bz = B.apply(z)
            xp = self.prox_f(A.apply(Bz - u + c, adjoint=True))

        with ops.name_scope("z_update"):
            Axp = A.apply(xp)
            zp = self.prox_g(B.apply(Axp + u - c, adjoint=True))

        with ops.name_scope("u_update"):
            Bzp = B.apply(zp)
            r = Axp - Bzp - c
            up = u + r

        with ops.name_scope("residuals"):
            def calculate_residuals():
                r_norm = linalg_ops.norm(r)
                s_norm = (
                    self.rho*linalg_ops.norm(A.apply(Bzp - Bz, adjoint=True)))

                eps_pri = (
                    atol*np.sqrt(self.p) +
                    rtol*math_ops.reduce_max([
                        linalg_ops.norm(Axp),
                        linalg_ops.norm(Bzp),
                        linalg_ops.norm(c)]))
                eps_dual = (
                    atol*np.sqrt(self.n) +
                    rtol*self.rho*linalg_ops.norm(A.apply(u, adjoint=True)))
                return [r_norm, s_norm, eps_pri, eps_dual]

            residuals = control_flow_ops.cond(
                need_residuals,
                calculate_residuals,
                self.zero_residuals)

        return [xp, zp, up], residuals

    def solve(self, max_iters=10000, epoch_iters=10, verbose=False, sess=None,
              atol=1e-4, rtol=1e-2, profile=False):
        t_start = time.time()
        tol = (rtol, atol)

        def cond(k, varz, residuals):
            return k < epoch_iters

        def body(k, varz, residuals):
            need_residuals = math_ops.equal(k, epoch_iters-1)
            varzp, residualsp = self.iterate(varz, tol, need_residuals)
            return [k+1, varzp, residualsp]

        loop_vars = [0, self.variables, self.zero_residuals()]
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
            if profile:
                run_options = config_pb2.RunOptions(
                    trace_level=config_pb2.RunOptions.FULL_TRACE)
                run_metadata = config_pb2.RunMetadata()
                _, r_norm0, s_norm0, eps_pri0, eps_dual0 = sess.run(
                    [epoch_op] + residuals_epoch,
                    options=run_options,
                    run_metadata=run_metadata)

                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open("/tmp/cvxflow_prox_admm_epoch_%d.json" % i, "w") as f:
                    f.write(ctf)
            else:
                _, r_norm0, s_norm0, eps_pri0, eps_dual0 = sess.run(
                    [epoch_op] + residuals_epoch)
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
