
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

        def A(x):
          return math_ops.matmul(A1, x)
        def AT(x):
          return math_ops.matmul(A1, x, transpose_a=True)

        prox_f = least_squares.LeastSquares(A=I, b=b)
        prox_g = absolute_value.AbsoluteValue(scale=lam)
        solver = pogs.POGS(prox_f, prox_g, (A, AT), shape=(m, n), dtype=dtype)
        x, _ = solver.solve(sess=sess)

        self.assertEqual(dtype, x.dtype)
        self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

  def testLasso(self):
    self._verify([[1.,-10],[1.,10.],[1.,0.]], [[2.],[2.],[2.]], 1,
                 [[1.6666666], [0]])

if __name__ == "__main__":
    test.main()
