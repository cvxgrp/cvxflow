
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops

from cvxflow.prox import admm
from cvxflow import conjugate_gradient


class POGS(object):
  """Proximal Operator Graph Solver.

  minimize    f(y) + g(x)
  subject to  y = Ax
  """
  def __init__(
      self, prox_f=None, prox_g=None, A=None, shape=None, dtype=dtypes.float32,
      name="POGS"):
    with ops.name_scope(name):
      self.prox_f = prox_f or (lambda x: x)
      self.prox_g = prox_g or (lambda x: x)
      self.A, self.AT = A
      self.m, self.n = shape
      self.admm = admm.ADMM(
        self._apply_prox, self._project, dtype=dtype, shape=(
          self.m+self.n, self.m+self.n, self.m+self.n, 1))

  def _split_y_x(self, y_x):
    return y_x[:self.m], y_x[self.m:]

  def _apply_prox(self, v):
    v_y, v_x = self._split_y_x(v)
    return array_ops.concat([self.prox_f(v_y), self.prox_g(v_x)], axis=0)

  def _project(self, v):
    v_y, v_x = self._split_y_x(v)

    # TODO(mwytock): use CGLS rather than CG here
    def I_ATA(x):
      return x + self.AT(self.A(x))
    x = conjugate_gradient.conjugate_gradient_solve(
      I_ATA,
      v_x + self.AT(v_y),
      v_x)[0]

    y = self.A(x)
    return array_ops.concat([y, x], axis=0)

  def solve(self, **kwargs):
    y_x, _, _ = self.admm.solve(**kwargs)
    return self._split_y_x(y_x)
