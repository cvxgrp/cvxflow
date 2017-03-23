
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.contrib import linalg

from cvxflow.prox import functions
from cvxflow.prox import solvers

class SolverTest(test.TestCase):
    pass

class ADMMLassoTest(SolverTest):
    def _verify(self, A, b, lam, expected_x):
        with self.test_session() as sess:
            prox_f = functions.LeastSquares(A=A, b=b)
            prox_g = functions.AbsoluteValue(scale=lam)
            A = linalg.LinearOperatorIdentity(num_rows=2)
            B = linalg.LinearOperatorDiag(
                constant_op.constant(-1., shape=(2,)))
            c = 0.

            solver = solvers.ADMM(prox_f, prox_g, A, B, c)
            solver.solve(sess=sess, verbose=False)
            self.assertAllClose(expected_x, sess.run(solver.x), atol=1e-4)

    def test_lasso(self):
        self._verify([[1.,2.],[3.,4.],[5.,6.]], [7.,8.,9.], 1,
                     [[-1.83334283], [3.20834076]])


class ADMMLeastAbsDevTest(SolverTest):
    def _verify(self, A, b, expected_x):
        with self.test_session() as sess:
            prox_f = functions.LeastSquares(W=A)
            prox_g = functions.AbsoluteValue()
            A = linalg.LinearOperatorMatrix(A)
            B = linalg.LinearOperatorDiag(
                constant_op.constant(-1., shape=(3,)))
            c = b

            solver = solvers.ADMM(prox_f, prox_g, A, B, c)
            solver.solve(sess=sess, verbose=False)
            self.assertAllClose(expected_x, sess.run(solver.x), atol=1e-4)

    def test_least_abs_dev(self):
        self._verify([[1.,1.],[1.,2.],[1.,3.]], [1.,2.,10.],
                     [[-3.5], [4.5]])


class ADMMQuantileRegression(SolverTest):
    def _verify(self, A, b, expected_x):
        with self.test_session() as sess:
            prox_f = functions.LeastSquares(W=A)
            prox_g = functions.AbsoluteValue(scale=(0.2,0.8))
            A = linalg.LinearOperatorMatrix(A)
            B = linalg.LinearOperatorDiag(
                constant_op.constant(-1., shape=(A.shape[0],)))
            c = b

            solver = solvers.ADMM(prox_f, prox_g, A, B, c)
            solver.solve(sess=sess, verbose=False)
            self.assertAllClose(expected_x, sess.run(solver.x), atol=1e-4)

    def test_quantile_regression(self):
        self._verify([[1.,1.],[1.,2.],[1.,3.],[1.,4.]], [1.,2.,10.,20.],
                     [[-14],  [8]])

class ADMMMultipleQuantileRegression(SolverTest):
    def _verify(self, A, b, expected_x):
        with self.test_session() as sess:
            prox_f = functions.LeastSquares(W=A, mu=1e-2)
            prox_g = functions.AbsoluteValue(scale=([0.2,0.5,0.8],[0.8,0.5,0.2]))
            A = linalg.LinearOperatorMatrix(A)
            B = linalg.LinearOperatorDiag(
                constant_op.constant(-1., shape=(A.shape[0],)))
            c = b

            solver = solvers.ADMM(prox_f, prox_g, A, B, c, num_columns=3)
            solver.solve(sess=sess, verbose=False)
            self.assertAllClose(expected_x, sess.run(solver.x), atol=1e-4)

    def test_multiple_quantile_regression(self):
        self._verify([[1.,1.],[1.,2.],[1.,3.],[1.,4.]], [1.,2.,10.,20.],
                     [[-14, -3.5, -5.33333], [8, 4.5, 6.33333]])



if __name__ == "__main__":
    test.main()
