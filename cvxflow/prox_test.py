
import tensorflow as tf

from cvxflow import prox

class ProxFunctionTest(tf.test.TestCase):
    @property
    def dtypes_to_test(self):
        return [tf.float32, tf.float64]

    def verify(self, v, expected_x, **kwargs):
        for dtype in self.dtypes_to_test:
            with self.test_session() as sess:
                if "dtype" in kwargs:
                    kwargs["dtype"] = dtype
                f = self.prox_function(**kwargs)
                x = sess.run(f(tf.convert_to_tensor(v, dtype=dtype)))
                self.assertEqual(dtype, x.dtype)
                self.assertAllClose(expected_x, x, atol=1e-5)

class AbsoluteValueTest(ProxFunctionTest):
    @property
    def prox_function(self):
        return prox.AbsoluteValue

    def testDefault(self):
        self.verify([-3.,-1.,4.], [-2.,0.,3.], scale=1.)
    def testScaleSymmetric(self):
        self.verify([-3.,-1.,4.], [-1.,0.,2.], scale=2.)
    def testScaleAsymmetric(self):
        self.verify([-4.,-2.,-1.,2.,3.], [-1.,0.,0.,1.,2.], scale=(3.,1.))
    def testScale2d(self):
        self.verify([[-3,-1,4.], [-3,-1,4.]],
                     [[0,0,3.], [-2,0,1.]],
                     scale=([[3],[1]],[[1],[3]]))

class LeastSquaresTest(ProxFunctionTest):
    @property
    def prox_function(self):
        return prox.LeastSquares

    def testBasic1d(self):
        self.verify(
            [[1.],[2.]], [[-0.853448], [2.448276]],
            A=[[1.,2.],[3.,4.],[5.,6.]],
            b=[[7.],[8.],[9.]],
            n=2,
            dtype=None)
    def testBasic2d(self):
        self.verify(
            [[1.,3.], [2., 4.]],
            [[-0.853448, -0.629311], [2.448276, 2.310345]],
            A=[[1.,2.],[3.,4.],[5.,6.]],
            b=[[7.],[8.],[9.]],
            n=2,
            dtype=None)
    def testBasic1dWithMu(self):
        self.verify(
            [[1.],[2.]], [[-0.728593], [2.347778]],
            A=[[1.,2.],[3.,4.],[5.,6.]],
            b=[[7.],[8.],[9.]],
            mu=0.1,
            n=2,
            dtype=None)
    def testConstrained1D(self):
        self.verify(
            [[1.],[2.]], [[ 1.07022], [2.16489]],
            C=[[1.,2.],[3.,4.],[5.,6.]],
            d=[[7.],[8.],[9.]],
            n=2,
            dtype=None)

class BlockSeparableTest(ProxFunctionTest):
    @property
    def prox_function(self):
        return prox.BlockSeparable

    def testBlockSeparable(self):
        self.verify([[-2], [2], [-2], [2]], [[-2], [2], [0], [2]],
                     proxs=[lambda x: x,
                            lambda x: tf.maximum(x, 0)],
                     shapes=[(1,2),(2,1)])

class NonnegativeTest(ProxFunctionTest):
    @property
    def prox_function(self):
        return prox.Nonnegative

    def testNonNegative(self):
        self.verify([1,-2,0], [1,0,0])


if __name__ == "__main__":
    tf.test.main()
