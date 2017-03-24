
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


def _get_size_and_type(A, W):
  if A is not None:
    graph_parents.extend(A.graph_parents)
    n = int(A.shape[1])
    dtype = A.dtype
    A = A.to_dense()
  elif W is not None:
    graph_parents.extend(W.graph_parents)
    n = int(W.shape[1])
    dtype = W.dtype
    W = W.to_dense()
  else:
    raise ValueError("Must specify either A or W")


def _get_normal_equation(A, W, mu, b):
  pass

class LeastSquares(prox_function.ProxFunction):
  """Least squares.

  (1/2)||Ax - b||^2 + (mu/2)||x||^2 + I(Cx = d) + (1/2)||x - v||^2
  """
  def __init__(self, A=None, b=None, C=None, d=None, W=None, mu=None,
               name="LeastSquares"):
    with ops.name_scope(name, values=[A, b, C, d, W, mu]):
      n, dtype = _compute_size_and_type(A, W)
      M, h = _get_normal_equation(A, W, mu, b)
      self.chol = math_ops.cholesky(M)

      if C:
        self.constrained = True
        eye = linalg_ops.eye(n, dtype=dtype)
        F = linalg_ops.cholesky_solve(C.apply(eye, adjoint=True))  # M^{-1}C'
        self.chol_constraint = math_ops.cholesky(C.apply(self.F))
        self.F = F
        self.C = C
        self.h = ...
      else:
        self.h = h
        self.constrained = False

      super(LeastSquares, self).__init__(
        graph_parents=[A, b, C, D, W, mu],
        name=name)


    def _call(self, v):
      if self.constrained:
        y = linalg_ops.cholesky_solve(
          self.chol_constraint, self.F.apply(v, adjoint=True))
        return linalg_ops.cholesky_solve(
          self.chol, self.h + self.C.apply(y, adjoint=True))
      else:
        return linalg_ops.cholesky_solve(self.chol, self.g + v)
