

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
import numpy as np

from cvxflow import conjugate_gradient

class ConjugateGradientTest(test.TestCase):
  @property
  def _dtypes_to_test(self):
    return [dtypes.float32, dtypes.float64]

  def _verify(self, A0, b0, x_init0):
    expected_x = np.linalg.solve(A0, b0)
    for dtype in self._dtypes_to_test:
      with self.test_session() as sess:
        A1 = ops.convert_to_tensor(A0, dtype=dtype)
        def A(x):
          return math_ops.matmul(A1, x)
        b = ops.convert_to_tensor(b0, dtype=dtype)
        x_init = ops.convert_to_tensor(x_init0, dtype=dtype)
        op = conjugate_gradient.cg_solve(A, b, x_init)
        x = sess.run(op)
        self.assertEqual(dtype, x.dtype)
        self.assertAllClose(expected_x, x)

  def testRandomATA(self):
    np.random.seed(0)
    n = 5
    A0 = np.random.randn(n,n)
    A0 = np.eye(n) + A0.T.dot(A0)
    b0 = np.random.randn(n,1)
    x_init0 = np.zeros((n,1))
    self._verify(A0, b0, x_init0)

  def testRandomATAInitSolution(self):
    np.random.seed(0)
    n = 5
    A0 = np.random.randn(n,n)
    A0 = np.eye(n) + A0.T.dot(A0)
    x0 = np.random.randn(n,1)
    b0 = A0.dot(x0)
    x_init0 = np.zeros((n,1))
    self._verify(A0, b0, x0)


class ConjugateGradientLSTest(test.TestCase):
  @property
  def _dtypes_to_test(self):
    return [dtypes.float32, dtypes.float64]

  def _verify(self, A0, b0, shift0, x_init0):
    expected_x = np.linalg.solve(
      A0.T.dot(A0) + shift0*np.eye(A0.shape[1]), A0.T.dot(b0))

    for dtype in self._dtypes_to_test:
      with self.test_session() as sess:
        A1 = ops.convert_to_tensor(A0, dtype=dtype)
        def A(x):
          return math_ops.matmul(A1, x)
        def AT(x):
          return math_ops.matmul(A1, x, transpose_a=True)
        b = ops.convert_to_tensor(b0, dtype=dtype)
        x_init = ops.convert_to_tensor(x_init0, dtype=dtype)
        shift = ops.convert_to_tensor(shift0, dtype=dtype)
        op = conjugate_gradient.cgls_solve(A, AT, b, x_init, shift=shift)
        x = sess.run(op)
        self.assertEqual(dtype, x.dtype)
        self.assertAllClose(expected_x, x)

  def testRandomA(self):
    np.random.seed(0)
    m = 10
    n = 5
    A = np.random.randn(m,n)
    b = np.random.randn(m,1)
    x_init = np.zeros((n,1))
    shift = 1
    self._verify(A, b, shift, x_init)

  def testShiftZero(self):
    np.random.seed(0)
    m = 10
    n = 5
    A = np.random.randn(m,n)
    b = np.random.randn(m,1)
    x_init = np.zeros((n,1))
    shift = 0
    self._verify(A, b, shift, x_init)

  def testInitSolution(self):
    np.random.seed(0)
    m = 10
    n = 5
    A = np.random.randn(m,n)
    x = np.random.randn(n,1)
    b = A.dot(x)
    shift = 0
    self._verify(A, b, shift, x)


if __name__ == "__main__":
    test.main()
