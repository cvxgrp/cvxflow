
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

from cvxflow import admm
from cvxflow import conjugate_gradient

PROJECT_TOL_MIN = 1e-2
PROJECT_TOL_MAX = 1e-2
PROJECT_TOL_POW = 1.3

class POGS(object):
  """Proximal Operator Graph Solver.

  minimize    f(y) + g(x)
  subject to  y = Ax
  """
  def __init__(
      self, prox_f=None, prox_g=None, A=None, shape=None, dtype=dtypes.float32,
      name=None):
    with ops.name_scope(name, "POGS"):
      self.prox_f = prox_f or (lambda x: x)
      self.prox_g = prox_g or (lambda x: x)
      self.A, self.AT = A
      self.m, self.n = shape
      self.admm = admm.ADMM(
        self._apply_prox, self._project, dtype=dtype, shape=(
          self.m+self.n, self.m+self.n, self.m+self.n, 1))

  def _split_y_x(self, y_x):
    return y_x[:self.m], y_x[self.m:]

  def _apply_prox(self, v, k):
    v_y, v_x = self._split_y_x(v)
    return array_ops.concat([self.prox_f(v_y), self.prox_g(v_x)], axis=0)

  def _project(self, v, k):
    v_y, v_x = self._split_y_x(v)

    k = math_ops.cast(k, v.dtype)
    tol = math_ops.maximum(
      PROJECT_TOL_MIN / math_ops.pow(k+1, PROJECT_TOL_POW),
      PROJECT_TOL_MAX)

    b = v_y - self.A(v_x)
    x_init = array_ops.zeros_like(v_x)
    x = conjugate_gradient.cgls_solve(
      self.A, self.AT, b, x_init, tol=tol, shift=1)
    x = x + v_x
    y = self.A(x)
    return array_ops.concat([y, x], axis=0)

  def solve(self, **kwargs):
    y_x, _, _ = self.admm.solve(**kwargs)
    return self._split_y_x(y_x)
