
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

from cvxflow.prox import prox_function


def _get_graph_parents(A, b, C, d, mu):
  graph_parents = []
  if A is not None:
    graph_parents.extend(A.graph_parents)
  if b is not None:
    graph_parents.append(b)
  if C is not None:
    graph_parents.extend(C.graph_parents)
  if d is not None:
    graph_parents.append(d)
  if mu is not None:
    graph_parents.append(mu)
  return graph_parents


def _get_normal_equation(A, mu, b, n, dtype):
  M = 0
  h = 0
  mu = mu if mu is not None else 0

  M = (mu+1)*linalg_ops.eye(n, dtype=dtype)
  if A is not None:
    M += A.apply(A.to_dense(), adjoint=True)
    if b is not None:
      h = A.apply(b, adjoint=True)

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
        n = int(A.shape[1])

      graph_parents = _get_graph_parents(A, b, C, d, mu)
      M, self.h = _get_normal_equation(A, mu, b, n, dtype)
      self.chol = linalg_ops.cholesky(M)

      if C:
        self.constrained = True
        self.C = C
        self.d = d

        # CM^{-1}C^T
        self.chol_constraint = C.apply(
          linalg_ops.cholesky_solve(
            self.chol,
            array_ops.transpose(C.to_dense())))

      else:
        self.constrained = False

      super(LeastSquares, self).__init__(
        graph_parents=graph_parents,
        name=name)

  def _call(self, v):
    if self.constrained:
      z = linalg_ops.cholesky_solve(self.chol, self.h + v)
      y = linalg_ops.cholesky_solve(
        self.chol_constraint, self.C.apply(z) - self.d)
      return linalg_ops.cholesky_solve(
        self.chol, self.h + v - self.C.apply(y, adjoint=True))
    else:
      return linalg_ops.cholesky_solve(self.chol, self.h + v)
