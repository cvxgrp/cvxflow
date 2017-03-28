
from tensorflow.contrib import linalg
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

from cvxflow.prox import absolute_value
from cvxflow.prox import least_squares
from cvxflow.prox import pogs

class POGSTest(test.TestCase):
  @property
  def _dtypes_to_test(self):
    return [dtypes.float32, dtypes.float64]


class LassoTest(POGSTest):
  def _verify(self, A0, b0, lam0, expected_x):
    for dtype in self._dtypes_to_test:
      with self.test_session() as sess:
        A1 = ops.convert_to_tensor(A0, dtype=dtype)
        m, n = A1.get_shape().as_list()
        b = ops.convert_to_tensor(b0, dtype=dtype)
        lam = ops.convert_to_tensor(lam0, dtype=dtype)
        I = linalg_ops.eye(m, dtype=dtype)

        def A((x,)):
          return [math_ops.matmul(A1, x)]
        def AT((x,)):
          return [math_ops.matmul(A1, x, transpose_a=True)]

        solver = pogs.POGS(
          prox_f=[least_squares.LeastSquares(A=I, b=b)],
          prox_g=[absolute_value.AbsoluteValue(scale=lam)],
          A=(A, AT),
          shape=([(m,1)], [(n,1)]),
          dtype=dtype)
        _, (x,) = solver.solve(sess=sess)

        self.assertEqual(dtype, x.dtype)
        self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

  def testLasso(self):
    self._verify([[1.,-10],[1.,10.],[1.,0.]], [[2.],[2.],[2.]], 1,
                 [[1.6666666], [0]])


def MultipleQuantileRegressionTest(POGSTest):
  def _verify(self, X0, y0, expected_theta):
    for dtype in self._dtypes_to_test:
      with self.test_session() as sess:
        X = ops.convert_to_tensor(X0, dtype=dtype)
        m, n = X.get_shape().as_list()
        y = ops.convert_to_tensor(y0, dtype=dtype)

        def A((theta,)):
          XT = math_ops.matmul(X, theta)
          return [XT, XT[:,1:] - XT[:,:-1]]

        def AT((W, Z)):
          XTW = math_ops.matmul(X, W, transpose_x=True)
          XTZ = math_ops.matmul(X, Z, transpose_x=True)
          return [XTW + (XTZ[1:,:] - XTZ[:-1,:])]

        scale = (ops.convert_to_tensor([0.2,0.5,0.8], dtype=dtype),
                 ops.convert_to_tensor([0.8,0.5,0.2], dtype=dtype))

        solver = pogs.POGS(
          prox_f=[
            composition.PreCompose(
              absolute_value.AbsoluteValue(scale=scale), b=-b),
            non_negative.NonNegative()],
          A=(A, AT),
          shape=([(m, k), (m, k-1)], [(n, k)]),
          dtype=dtype)
        _, (theta,) = solver.solve(sess=sess)

        self.assertEqual(dtype, theta.dtype)
        self.assertAllClose(expected_theta, theta, rtol=1e-2, atol=1e-4)


if __name__ == "__main__":
    test.main()
