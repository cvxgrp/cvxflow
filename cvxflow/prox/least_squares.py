
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

def _solve(v):
  # for x
  D_chol = linalg_ops.cholesky(D)

  return x




class LeastSquares(prox_function.ProxFunction):
  """Least squares.

  (1/2)||Ax - b||^2 + (mu/2)||x||^2 + (1/2)||Wx - v||^2 + I(Gx = h)
  """
  def __init__(self,
               A=None,
               b=None,
               mu=None,
               W=None,
               G=None,
               h=None,
               name="LeastSquares"):
    with ops.name_scope(name, values=[A, b, mu, W, G, h]):
      # Solve the 2x2 system
      #
      # [D G'][x] = [W'v + A'b]
      # [G   ][y]   [h]
      #
      # D = A'A + mu*I + W'W
      D_chol = linalg_ops.cholesky(D)

      if G is not None:
        GDinvGT = math_ops.matmul(
          linalg_ops.cholesky_solve(D_chol, array_ops.transpose(G)))
        GDinvGT_chol = linalg_ops.cholesky(GDinvGT)







      n = None
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

            if b is not None:
                b = ops.convert_to_tensor(b)
                graph_parents.append(b)
            if mu is not None:
                mu = ops.convert_to_tensor(mu)
                graph_parents.append(mu)


            M = 0
            if A is not None:
                M += math_ops.matmul(A, A, transpose_a=True)
            if A is not None and b is not None:
                self.ATb = math_ops.matmul(A, b, transpose_a=True)
            else:
                self.ATb = 0

            if W is not None:
                M += math_ops.matmul(W, W, transpose_a=True)
            else:
                M += linalg_ops.eye(n, dtype=dtype)

            if mu is not None:
                M += mu*linalg_ops.eye(n, dtype=dtype)

            self.L = linalg_ops.cholesky(M)

      super(LeastSquares, self).__init__(
        graph_parents=[A, b, mu, W, G, h],
        name=name)


    def _call(self, v):
        return linalg_ops.cholesky_solve(self.L, self.ATb + v)
