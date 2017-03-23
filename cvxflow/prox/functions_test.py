

from tensorflow.python.framework import ops
from tensorflow.python.platform import test

from cvxflow.prox import functions

class ProxFunctionTest(test.TestCase):
    def _verify(self, v, expected_x, **kwargs):
        with self.test_session() as sess:
            f = self.__class__.PROX_FUNCTION(**kwargs)
            x = sess.run(f(ops.convert_to_tensor(v)))
            self.assertAllClose(expected_x, x, atol=1e-5)

class AbsoluteValueTest(ProxFunctionTest):
    PROX_FUNCTION = functions.AbsoluteValue
    def testDefault(self):
        self._verify([-3.,-1.,4.], [-2.,0.,3.])
    def testScaleSymmetric(self):
        self._verify([-3.,-1.,4.], [-1.,0.,2.], scale=2)
    def testScaleAsymmetric(self):
        self._verify([-4.,-2.,-1.,2.,3.], [-1.,0.,0.,1.,2.], scale=(3,1))
    def testScale2d(self):
        self._verify([[-3,-1,4.], [-3,-1,4.]],
                     [[0,0,3.], [-2,0,1.]],
                     scale=([[3],[1]],[[1],[3]]))

class LeastSquaresTest(ProxFunctionTest):
    PROX_FUNCTION = functions.LeastSquares
    def testBasic(self):
        self._verify([1.,2.], [-0.853448, 2.448276],
                     A=[[1.,2.],[3.,4.],[5.,6.]], b=[7.,8.,9.])
    def testBasic2d(self):
        self._verify([[1.,3.], [2., 4.]],
                     [[-0.853448, -0.629311], [2.448276, 2.310345]],
                     A=[[1.,2.],[3.,4.],[5.,6.]], b=[7.,8.,9.])
    def testBasicMu(self):
        self._verify([1.,2.], [-0.728593, 2.347778],
                     A=[[1.,2.],[3.,4.],[5.,6.]], b=[7.,8.,9.], mu=0.1)
        pass

if __name__ == "__main__":
    test.main()
