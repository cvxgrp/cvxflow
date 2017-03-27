
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

class ADMM(object):
    """Alternating direction method of multipliers.

    minimize    f(x) + g(z)
    subject to  Ax + Bz = c
    """
    def __init__(
            self, argmin_f, argmin_g, A=None, B=None, c=None, rho=1,
            shape=None, dtype=dtypes.float32):
        if not shape:
            raise ValueError("Must specify shape")

        self.argmin_f = argmin_f
        self.argmin_g = argmin_g
        self.A = A or (lambda x: x, lambda x: x)
        self.B = B or (lambda x: -x, lambda x: -x)
        self.c = constant_op.constant(0, dtype=dtype) if c is None else c
        self.rho = rho

        self.x = variables.Variable(
            array_ops.zeros(shape=(shape[0],shape[3]), dtype=dtype))
        self.z = variables.Variable(
            array_ops.zeros(shape=(shape[1],shape[3]), dtype=dtype))
        self.u = variables.Variable(
            array_ops.zeros(shape=(shape[2],shape[3]), dtype=dtype))
        self.variables = [self.x, self.z, self.u]

        # Variable sizes are used in computing primal/dual epsilon
        self.n, self.m, self.p = [
            np.prod(x.get_shape().as_list()) for x in self.variables]

        self.default_residuals = [
            constant_op.constant(0, dtype=dtype) for _ in range(4)]

    def _iterate(self, (x, z, u), (rtol, atol), need_residuals):
        A, AT = self.A
        B, BT = self.B
        c = self.c

        with ops.name_scope("x_update"):
            Bz = B(z)
            xp = self.argmin_f(AT(Bz - u + c))

        with ops.name_scope("z_update"):
            Axp = A(xp)
            zp = self.argmin_g(B(Axp + u - c))

        with ops.name_scope("u_update"):
            Bzp = B(zp)
            r = Axp - Bzp - c
            up = u + r

        with ops.name_scope("residuals"):
            def calculate_residuals():
                r_norm = linalg_ops.norm(r)
                s_norm = (
                    self.rho*linalg_ops.norm(AT(Bzp - Bz)))
                eps_pri = (
                    atol*np.sqrt(self.p) +
                    rtol*math_ops.reduce_max([
                        linalg_ops.norm(Axp),
                        linalg_ops.norm(Bzp),
                        linalg_ops.norm(c)]))
                eps_dual = (
                    atol*np.sqrt(self.n) +
                    rtol*self.rho*linalg_ops.norm(AT(u)))
                return [r_norm, s_norm, eps_pri, eps_dual]

            residuals = control_flow_ops.cond(
                need_residuals,
                calculate_residuals,
                lambda: self.default_residuals)

        return [xp, zp, up], residuals

    @property
    def _output_variables(self):
        return self.variables

    def solve(self, max_iters=10000, epoch_iters=10, verbose=False, sess=None,
              atol=1e-4, rtol=1e-2, profile=False):
        t_start = time.time()
        tol = (rtol, atol)

        def cond(k, varz, residuals):
            return k < epoch_iters

        def body(k, varz, residuals):
            need_residuals = math_ops.equal(k, epoch_iters-1)
            varzp, residualsp = self._iterate(varz, tol, need_residuals)
            return [k+1, varzp, residualsp]

        loop_vars = [0, self.variables, self.default_residuals]
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

        return sess.run(self._output_variables)
