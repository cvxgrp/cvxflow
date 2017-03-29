
import numpy as np

from tensorflow.contrib import linalg
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

from cvxflow import block_ops
from cvxflow import pogs
from cvxflow import prox

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
        def AT(y):
          return math_ops.matmul(A1, y, transpose_a=True)

        solver = pogs.POGS(
          prox_f=prox.LeastSquares(A=I, b=b, n=m, dtype=dtype),
          prox_g=prox.AbsoluteValue(scale=lam),
          A=(A, AT),
          shape=(m, n),
          dtype=dtype)
        _, x = solver.solve(sess=sess)

        self.assertEqual(dtype, x.dtype)
        self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

  def testLasso(self):
    self._verify([[1.,-10],[1.,10.],[1.,0.]], [[2.],[2.],[2.]], 1,
                 [[1.6666666], [0]])


class MultipleQuantileRegressionTest(POGSTest):
  def _verify(self, X0, y0, expected_obj_val):
    for dtype in self._dtypes_to_test:
      with self.test_session() as sess:
        tau = np.array([0.2,0.5,0.8])
        X = ops.convert_to_tensor(X0, dtype=dtype)
        y = ops.convert_to_tensor(y0, dtype=dtype)
        m, n = X.get_shape().as_list()
        k = len(tau)

        # Linear system is
        # W = X*theta
        # Z = X*theta*D
        # where D is the difference operator.
        x_shape = [(n,k)]
        y_shape = [(m,k), (m,k-1)]
        x_slices = block_ops.get_slices(x_shape)
        y_slices = block_ops.get_slices(y_shape)
        def A(x):
          theta, = block_ops.to_list(x, x_slices)
          XT = math_ops.matmul(X, theta)
          XTD = XT[:,1:] - XT[:,:-1]
          return block_ops.to_vector([XT, XTD])
        def AT(y):
          W, Z = block_ops.to_list(y, y_slices)
          XTW = math_ops.matmul(X, W, transpose_a=True)
          XTZ = math_ops.matmul(X, Z, transpose_a=True)
          XTZDT = array_ops.concat(
            [-XTZ[:,:1], XTZ[:,:-1] - XTZ[:,1:], XTZ[:,-1:]], axis=1)
          return block_ops.to_vector([XTW + XTZDT])

        scale = (ops.convert_to_tensor(tau, dtype=dtype),
                 ops.convert_to_tensor(1-tau, dtype=dtype))
        tilted_l1 = prox.AbsoluteValue(scale=scale)
        def prox_quantile_loss(v):
          return tilted_l1(v-y) + y

        solver = pogs.POGS(
          prox_f=prox.BlockSeparable(
            proxs=[prox_quantile_loss, prox.Nonnegative()],
            shapes=y_shape),
          prox_g=prox.LeastSquares(mu=1e-2, n=n*k, dtype=dtype),
          A=(A, AT),
          shape=(m*k+m*(k-1), n*k),
          dtype=dtype)
        _, theta = solver.solve(sess=sess)
        theta = theta.reshape(n, k)

        z = np.array(X0).dot(theta) - np.array(y0)
        obj_val = (np.sum(-tau*np.minimum(z, 0) + (1-tau)*np.maximum(z, 0)) +
                   1e-2/2*np.sum(theta**2))

        self.assertEqual(dtype, theta.dtype)
        self.assertAllClose(expected_obj_val, obj_val, rtol=1e-2, atol=1e-4)

  def testMultipleQuantileRegression(self):
    self._verify([[1.,1.],[1.,2.],[1.,3.],[1.,4.]], [[1.],[2.],[10.],[20.]],
                 10.0051362724)


if __name__ == "__main__":
    test.main()
