
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

from cvxflow.prox import prox_function


def _get_normal_equation(A, mu, b, n, dtype):
  M = 0
  h = 0
  mu = mu if mu is not None else 0

  M = (mu+1)*linalg_ops.eye(n, dtype=dtype)
  if A is not None:
    M += math_ops.matmul(A, A, transpose_a=True)
    if b is not None:
      h = math_ops.matmul(A, b, transpose_a=True)

  return M, h


class LeastSquares(prox_function.ProxFunction):
  """Least squares.

  (1/2)||Ax - b||^2 + (mu/2)||x||^2 + I(Cx = d) + (1/2)||x - v||^2
  """
  def __init__(self, A=None, b=None, C=None, d=None, mu=None, n=None, dtype=None,
               name="LeastSquares"):
    with ops.name_scope(name, values=[A, b, C, d, mu]):
      if A is None:
        if n is None or dtype is None:
          raise ValueError("Must specify A or both dtype and n")
        dtype = dtype
        n = n
      else:
        dtype = A.dtype
        n = int(A.get_shape()[1])

      M, self.h = _get_normal_equation(A, mu, b, n, dtype)
      self.chol = linalg_ops.cholesky(M)

      if C is not None:
        self.constrained = True
        self.C = C
        self.CT = array_ops.transpose(C)
        self.d = d
        self.chol_constraint = math_ops.matmul(self.C,
          linalg_ops.cholesky_solve(self.chol, self.CT))  # CM^{-1}C^T
      else:
        self.constrained = False

      super(LeastSquares, self).__init__(
        graph_parents=[A, b, C, d, mu],
        name=name)

  def _call(self, v):
    if self.constrained:
      z = linalg_ops.cholesky_solve(self.chol, self.h + v)
      y = linalg_ops.cholesky_solve(
        self.chol_constraint, math_ops.matmul(self.C, z) - self.d)
      return linalg_ops.cholesky_solve(
        self.chol, self.h + v - math_ops.matmul(self.CT, y))
    else:
      return linalg_ops.cholesky_solve(self.chol, self.h + v)
