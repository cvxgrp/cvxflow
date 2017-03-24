

from tensorflow.contrib import linalg
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

from cvxflow.prox import functions

def np_cholesky_solve(L, b):
    return np.linalg.solve(L.T, np.linalg.solve(L, b))

def np_solve(A, b, C, d, W, v, mu):
    # [A'A + W'W + mu*I  C'][x] = [A'b + W'v]
    # [C                 0 ][y]   [d]
    I = np.eye(A.shape[1])
    L1 = np.linalg.cholesky(A.T.dot(A) + W.T.dot(W) + mu*I)
    F = np_cholesky_solve(L1, C.T)
    L2 = C.dot(C.dot(F))
    g = A.T.dot(b) + W.T.dot(v)
    y = np_cholesky_solve(L2, F.T.dot(g) - d)
    x = np_cholesky_solve(L1, g - C.T.dot(y))
    return x



class LeastSquaresTest(test.TestCase):
    @property
    def _dtypes_to_test(self):
        return [dtypes.float32, dtypes.float64]

    def

    def _convert_inputs(self, dtype, kwargs):
        retval = {}
        for k, v in kwargs.items():
            if isinstance(v, tuple):
                v = tuple(ops.convert_to_tensor(x, dtype=dtype) for x in v)
            elif isinstance(v, linalg.LinearOperatorMatrix):
                v = linalg.LinearOperatorMatrix(
                    math_ops.cast(v.to_dense(), dtype))
            else:
                v = ops.convert_to_tensor(v, dtype=dtype)
            retval[k] = v
        return retval

    def _verify(self, v, expected_x, **kwargs):
        for dtype in self._dtypes_to_test:
            with self.test_session() as sess:
                kwargs_dtype = self._convert_inputs(dtype, kwargs)
                f = self._prox_function(**kwargs_dtype)
                x = sess.run(f(ops.convert_to_tensor(v, dtype=dtype)))
                self.assertEqual(dtype, x.dtype)
                self.assertAllClose(expected_x, x, atol=1e-5)

class AbsoluteValueTest(ProxFunctionTest):
    @property
    def _prox_function(self):
        return functions.AbsoluteValue

    def testDefault(self):
        self._verify([-3.,-1.,4.], [-2.,0.,3.], scale=1.)
    def testScaleSymmetric(self):
        self._verify([-3.,-1.,4.], [-1.,0.,2.], scale=2.)
    def testScaleAsymmetric(self):
        self._verify([-4.,-2.,-1.,2.,3.], [-1.,0.,0.,1.,2.], scale=(3.,1.))
    def testScale2d(self):
        self._verify([[-3,-1,4.], [-3,-1,4.]],
                     [[0,0,3.], [-2,0,1.]],
                     scale=([[3],[1]],[[1],[3]]))

class LeastSquaresTest(ProxFunctionTest):
    @property
    def _prox_function(self):
        return functions.LeastSquares

    def testBasic1d(self):
        self._verify(
            [[1.],[2.]], [[-0.853448], [2.448276]],
            A=linalg.LinearOperatorMatrix([[1.,2.],[3.,4.],[5.,6.]]),
            b=[[7.],[8.],[9.]])
    def testBasic2d(self):
        self._verify(
            [[1.,3.], [2., 4.]],
            [[-0.853448, -0.629311], [2.448276, 2.310345]],
            A=linalg.LinearOperatorMatrix([[1.,2.],[3.,4.],[5.,6.]]),
            b=[[7.],[8.],[9.]])
    def testBasic1dWithMu(self):
        self._verify(
            [[1.],[2.]], [[-0.728593], [2.347778]],
            A=linalg.LinearOperatorMatrix([[1.,2.],[3.,4.],[5.,6.]]),
            b=[[7.],[8.],[9.]],
            mu=0.1)

if __name__ == "__main__":
    test.main()
