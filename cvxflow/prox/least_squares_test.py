

from tensorflow.python.framework import ops
from tensorflow.python.platform import test

from cvxflow.prox import least_squares
from cvxflow.prox import prox_function_testutil


class LeastSquaresTest(prox_function_testutil.ProxFunctionTest):
  def _get_prox_function_for_dtype(
      self, dtype, A=None, b=None, C=None, d=None, mu=None, n=None):
    A = ops.convert_to_tensor(A, dtype=dtype) if A else None
    b = ops.convert_to_tensor(b, dtype=dtype) if b else None
    C = ops.convert_to_tensor(C, dtype=dtype) if C else None
    d = ops.convert_to_tensor(d, dtype=dtype) if d else None
    mu = ops.convert_to_tensor(mu, dtype=dtype) if mu else None
    return least_squares.LeastSquares(
      A=A, b=b, C=C, d=d, mu=mu, dtype=dtype, n=n)

  def testBasic1d(self):
    self._verify(
      [[1.],[2.]], [[-0.853448], [2.448276]],
      A=[[1.,2.],[3.,4.],[5.,6.]],
      b=[[7.],[8.],[9.]],
      n=2)
  def testBasic2d(self):
    self._verify(
      [[1.,3.], [2., 4.]],
      [[-0.853448, -0.629311], [2.448276, 2.310345]],
      A=[[1.,2.],[3.,4.],[5.,6.]],
      b=[[7.],[8.],[9.]],
      n=2)
  def testBasic1dWithMu(self):
    self._verify(
      [[1.],[2.]], [[-0.728593], [2.347778]],
      A=[[1.,2.],[3.,4.],[5.,6.]],
      b=[[7.],[8.],[9.]],
      mu=0.1,
      n=2)
  def testConstrained1D(self):
    self._verify(
      [[1.],[2.]], [[ 1.07022], [2.16489]],
      C=[[1.,2.],[3.,4.],[5.,6.]],
      d=[[7.],[8.],[9.]],
      n=2)

# TODO(mwytock): Add some random tests using np_solve()
#class RandomLeastSquaresTest(prox_function_testutil.ProxFunctionTest):
  # def np_cholesky_solve(L, b):
  #   return np.linalg.solve(L.T, np.linalg.solve(L, b))

  # def np_solve(A, b, C, d, W, v, mu):
  #   # [A'A + W'W + mu*I  C'][x] = [A'b + W'v]
  #   # [C                 0 ][y]   [d]
  #   I = np.eye(A.shape[1])
  #   L1 = np.linalg.cholesky(A.T.dot(A) + W.T.dot(W) + mu*I)
  #   F = np_cholesky_solve(L1, C.T)
  #   L2 = C.dot(C.dot(F))
  #   g = A.T.dot(b) + W.T.dot(v)
  #   y = np_cholesky_solve(L2, F.T.dot(g) - d)
  #   x = np_cholesky_solve(L1, g - C.T.dot(y))
  #   return x

if __name__ == "__main__":
  test.main()
