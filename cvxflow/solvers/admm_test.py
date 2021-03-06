
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from cvxflow import prox
from cvxflow.solvers import admm

def argmin_prox(prox):
  def argmin(v, k):
    return prox(v)
  return argmin


class ADMMTest(test.TestCase):
  @property
  def _dtypes_to_test(self):
    return [dtypes.float32, dtypes.float64]


class LassoTest(ADMMTest):
  def _verify(self, A, b, lam, expected_x):
    for dtype in self._dtypes_to_test:
      with self.test_session() as sess:
        n = len(A[0])
        argmin_f = argmin_prox(prox.LeastSquares(A=A, b=b, n=n, dtype=dtype))
        argmin_g = argmin_prox(prox.AbsoluteValue(scale=lam))
        solver = admm.ADMM(argmin_f, argmin_g, shape=(n,n,n,1), dtype=dtype)
        x, _, _ = solver.solve(sess=sess)

        self.assertEqual(dtype, x.dtype)
        self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

  def testLasso(self):
    self._verify([[1.,-10],[1.,10.],[1.,0.]], [[2.],[2.],[2.]], 1,
                 [[1.6666666], [0]])


# class LeastAbsDevTest(ADMMTest):
#   def _verify(self, A0, b0, expected_x):
#     for dtype in self._dtypes_to_test:
#       with self.test_session() as sess:
#         A = linalg.LinearOperatorMatrix(
#           ops.convert_to_tensor(A0, dtype=dtype))
#         b = ops.convert_to_tensor(b0, dtype=dtype)
#         scale = ops.convert_to_tensor(1, dtype=dtype)
#         I = linalg.LinearOperatorIdentity(num_rows=A.shape[0], dtype=dtype)

#         argmin_f = functions.LeastSquares(W=A)
#         argmin_g = functions.AbsoluteValue(scale)
#         A = A
#         B = I
#         c = b
#         solver = solvers.ADMM(argmin_f, argmin_g, A, B, c)
#         x, _, _ = solver.solve(sess=sess)

#         self.assertEqual(dtype, x.dtype)
#         self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

#   def testBasic(self):
#     self._verify([[1.,1.],[1.,2.],[1.,3.]], [[1.],[2.],[10.]],
#                  [[-3.5], [4.5]])

# class QuantileRegressionTest(ADMMTest):
#   def _verify(self, A0, b0, expected_x):
#     for dtype in self._dtypes_to_test:
#       with self.test_session() as sess:
#         A = linalg.LinearOperatorMatrix(
#           ops.convert_to_tensor(A0, dtype=dtype))
#         b = ops.convert_to_tensor(b0, dtype=dtype)
#         scale = (ops.convert_to_tensor(0.2, dtype=dtype),
#                  ops.convert_to_tensor(0.8, dtype=dtype))
#         I = linalg.LinearOperatorIdentity(num_rows=A.shape[0], dtype=dtype)
#         mu = ops.convert_to_tensor(1e-2, dtype=dtype)

#         argmin_f = functions.LeastSquares(W=A, mu=mu)
#         argmin_g = functions.AbsoluteValue(scale)
#         A = A
#         B = I
#         c = b
#         solver = solvers.ADMM(argmin_f, argmin_g, A, B, c)
#         x, _, _ = solver.solve(sess=sess)

#         self.assertEqual(dtype, x.dtype)
#         self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

#   def testBasic(self):
#     self._verify([[1.,1.],[1.,2.],[1.,3.],[1.,4.]], [[1.],[2.],[10.],[20.]],
#                  [[-14],  [8]])


# class MultipleQuantileRegressionTest(ADMMTest):
#   def _verify(self, A0, b0, expected_x):
#     for dtype in self._dtypes_to_test:
#       with self.test_session() as sess:
#         A = linalg.LinearOperatorMatrix(
#           ops.convert_to_tensor(A0, dtype=dtype))
#         b = ops.convert_to_tensor(b0, dtype=dtype)
#         scale = (ops.convert_to_tensor([0.2,0.5,0.8], dtype=dtype),
#                  ops.convert_to_tensor([0.8,0.5,0.2], dtype=dtype))
#         I = linalg.LinearOperatorIdentity(num_rows=A.shape[0], dtype=dtype)
#         mu = ops.convert_to_tensor(1e-2, dtype=dtype)

#         argmin_f = functions.LeastSquares(W=A, mu=mu)
#         argmin_g = functions.AbsoluteValue(scale)
#         A = A
#         B = I
#         c = b
#         solver = solvers.ADMM(argmin_f, argmin_g, A, B, c, num_columns=3)
#         x, _, _ = solver.solve(sess=sess)

#         self.assertEqual(dtype, x.dtype)
#         self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

#   def testBasic(self):
#     self._verify([[1.,1.],[1.,2.],[1.,3.],[1.,4.]], [[1.],[2.],[10.],[20.]],
#                  [[-14, -3.5, -5.33333], [8, 4.5, 6.33333]])


if __name__ == "__main__":
  test.main()
