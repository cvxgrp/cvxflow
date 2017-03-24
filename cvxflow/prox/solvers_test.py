
from tensorflow.contrib import linalg
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from cvxflow.prox import functions
from cvxflow.prox import solvers

class SolverTest(test.TestCase):
    @property
    def _dtypes_to_test(self):
        return [dtypes.float32, dtypes.float64]


class LassoTest(SolverTest):
    def _verify(self, A0, b0, lam0, expected_x):
        for dtype in self._dtypes_to_test:
            with self.test_session() as sess:
                A = linalg.LinearOperatorMatrix(
                    ops.convert_to_tensor(A0, dtype=dtype))
                b = ops.convert_to_tensor(b0, dtype=dtype)
                lam = ops.convert_to_tensor(lam0, dtype=dtype)
                I = linalg.LinearOperatorIdentity(num_rows=A.shape[1], dtype=dtype)

                prox_f = functions.LeastSquares(A=A, b=b)
                prox_g = functions.AbsoluteValue(scale=lam)
                A = I
                B = I
                solver = solvers.ADMM(prox_f, prox_g, A, B)
                x, _, _ = solver.solve(sess=sess)

            self.assertEqual(dtype, x.dtype)
            self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

    def testBasic(self):
        self._verify([[1.,-10],[1.,10.],[1.,0.]], [[2.],[2.],[2.]], 1,
                     [[1.6666666], [0]])


class LeastAbsDevTest(SolverTest):
    def _verify(self, A0, b0, expected_x):
        for dtype in self._dtypes_to_test:
            with self.test_session() as sess:
                A = linalg.LinearOperatorMatrix(
                    ops.convert_to_tensor(A0, dtype=dtype))
                b = ops.convert_to_tensor(b0, dtype=dtype)
                scale = ops.convert_to_tensor(1, dtype=dtype)
                I = linalg.LinearOperatorIdentity(num_rows=A.shape[0], dtype=dtype)

                prox_f = functions.LeastSquares(W=A)
                prox_g = functions.AbsoluteValue(scale)
                A = A
                B = I
                c = b
                solver = solvers.ADMM(prox_f, prox_g, A, B, c)
                x, _, _ = solver.solve(sess=sess)

            self.assertEqual(dtype, x.dtype)
            self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

    def testBasic(self):
        self._verify([[1.,1.],[1.,2.],[1.,3.]], [[1.],[2.],[10.]],
                     [[-3.5], [4.5]])

class QuantileRegressionTest(SolverTest):
    def _verify(self, A0, b0, expected_x):
        for dtype in self._dtypes_to_test:
            with self.test_session() as sess:
                A = linalg.LinearOperatorMatrix(
                    ops.convert_to_tensor(A0, dtype=dtype))
                b = ops.convert_to_tensor(b0, dtype=dtype)
                scale = (ops.convert_to_tensor(0.2, dtype=dtype),
                         ops.convert_to_tensor(0.8, dtype=dtype))
                I = linalg.LinearOperatorIdentity(num_rows=A.shape[0], dtype=dtype)
                mu = ops.convert_to_tensor(1e-2, dtype=dtype)

                prox_f = functions.LeastSquares(W=A, mu=mu)
                prox_g = functions.AbsoluteValue(scale)
                A = A
                B = I
                c = b
                solver = solvers.ADMM(prox_f, prox_g, A, B, c)
                x, _, _ = solver.solve(sess=sess)

            self.assertEqual(dtype, x.dtype)
            self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

    def testBasic(self):
        self._verify([[1.,1.],[1.,2.],[1.,3.],[1.,4.]], [[1.],[2.],[10.],[20.]],
                     [[-14],  [8]])


class MultipleQuantileRegressionTest(SolverTest):
    def _verify(self, A0, b0, expected_x):
        for dtype in self._dtypes_to_test:
            with self.test_session() as sess:
                A = linalg.LinearOperatorMatrix(
                    ops.convert_to_tensor(A0, dtype=dtype))
                b = ops.convert_to_tensor(b0, dtype=dtype)
                scale = (ops.convert_to_tensor([0.2,0.5,0.8], dtype=dtype),
                         ops.convert_to_tensor([0.8,0.5,0.2], dtype=dtype))
                I = linalg.LinearOperatorIdentity(num_rows=A.shape[0], dtype=dtype)
                mu = ops.convert_to_tensor(1e-2, dtype=dtype)

                prox_f = functions.LeastSquares(W=A, mu=mu)
                prox_g = functions.AbsoluteValue(scale)
                A = A
                B = I
                c = b
                solver = solvers.ADMM(prox_f, prox_g, A, B, c, num_columns=3)
                x, _, _ = solver.solve(sess=sess)

            self.assertEqual(dtype, x.dtype)
            self.assertAllClose(expected_x, x, rtol=1e-2, atol=1e-4)

    def testBasic(self):
        self._verify([[1.,1.],[1.,2.],[1.,3.],[1.,4.]], [[1.],[2.],[10.],[20.]],
                     [[-14, -3.5, -5.33333], [8, 4.5, 6.33333]])


if __name__ == "__main__":
    test.main()
