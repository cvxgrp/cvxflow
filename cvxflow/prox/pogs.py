
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops

from cvxflow.prox import admm

class POGS(admm.ADMM):
  """Proximal Operator Graph Solver.

  minimize    f(y) + g(x)
  subject to  y = Ax
  """
  def __init__(
      self, prox_f=None, prox_g=None, A=None, shape=None, dtype=dtypes.float32,
      name="POGS"):
    with ops.name_scope(name):
      m, n = shape

      prox_f = prox_f or (lambda x: x)
      prox_g = prox_g or (lambda x: x)
      def argmin_f(v):
        return array_ops.concat([prox_f(v[:m]), prox_g(v[m:])], axis=0)

      A, AT = A
      eye = linalg_ops.eye(n, dtype=dtype)
      chol = linalg_ops.cholesky(AT(A(eye)) + eye)
      def argmin_g(v):
        d, c = v[:m], v[m:]
        x = linalg_ops.cholesky_solve(chol, c + AT(d))
        y = A(x)
        return array_ops.concat([y, x], axis=0)

      admm_n = m + n
      self._pogs_m = m
      super(POGS, self).__init__(
        argmin_f, argmin_g, dtype=dtype, shape=(admm_n, admm_n, admm_n, 1))

  @property
  def _output_variables(self):
    return self.x[self._pogs_m:], self.x[:self._pogs_m]
