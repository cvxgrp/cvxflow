
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import linalg
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

class Solver(object):
    def solve(self, max_iters=1000, epoch_iters=10, verbose=True, sess=None):
        def cond(*args):
            k, _ = args[0], args[1:]
            return k < epoch_iters

        def body(*args):
            k, state = args[0], args[1:]
            next_state = list(self.iterate(*state))
            return [k+1] + next_state

        loop_vars = [0] + self.variables
        values = control_flow_ops.while_loop(cond, body, loop_vars)[1:]
        init_op = variables.global_variables_initializer()
        epoch_op = control_flow_ops.group(
            *[x.assign(val) for x, val in zip(self.variables, values)])

        if sess is None:
            sess = session.Session()

        sess.run(init_op)
        num_epochs = max_iters // epoch_iters
        for i in range(num_epochs):
            sess.run(epoch_op)
            if verbose:
                print("iter " + str((i+1)*epoch_iters))


class ADMM(Solver):
    """Alternating direction method of multipliers.

    minimize    f(x) + g(z)
    subject to  Ax + Bz = c
    """
    def __init__(self, prox_f, prox_g, A, B, c, num_columns=1):
        self.prox_f = prox_f
        self.prox_g = prox_g
        self.A = A
        self.B = B
        self.c = ops.convert_to_tensor(c)

        if len(self.c.get_shape()) == 0:
            self.c = array_ops.reshape(c, (1,1))
        elif len(self.c.get_shape()) == 1:
            self.c = array_ops.reshape(c, (int(self.c.get_shape()[0]), 1))

        m, n = A.shape
        p = B.shape[1]

        self.x = variables.Variable(array_ops.zeros(shape=(n,num_columns)))
        self.z = variables.Variable(array_ops.zeros(shape=(p,num_columns)))
        self.u = variables.Variable(array_ops.zeros(shape=(m,num_columns)))
        self.variables = [self.x, self.z, self.u]

    def iterate(self, x, z, u):
        A, B, c = self.A, self.B, self.c
        prox_f, prox_g = self.prox_f, self.prox_g
        x = prox_f(A.apply(c - B.apply(z) - u, adjoint=True))
        z = prox_g(B.apply(c - A.apply(x) - u, adjoint=True))
        u += A.apply(x) + B.apply(z) - c
        return x, z, u
