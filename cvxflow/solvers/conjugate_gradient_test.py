
import tensorflow as tf
import numpy as np

from cvxflow.solvers import conjugate_gradient

class ConjugateGradientTest(tf.test.TestCase):
    @property
    def dtypes_to_test(self):
        return [tf.float32, tf.float64]

    def verify(self, A0, b0, x_init0):
        expected_x = np.linalg.solve(A0, b0)
        for dtype in self.dtypes_to_test:
            with self.test_session() as sess:
                A1 = tf.convert_to_tensor(A0, dtype=dtype)
                def A(x):
                    return tf.matmul(A1, x)
                b = tf.convert_to_tensor(b0, dtype=dtype)
                x_init = tf.convert_to_tensor(x_init0, dtype=dtype)

                cg = conjugate_gradient.ConjugateGradient(A, b, x_init)
                state = cg.solve()
                self.assertEqual(dtype, state.x.dtype)

                state_np = sess.run(state)
                self.assertLess(state_np.k, cg.max_iterations)
                self.assertAllClose(expected_x, state_np.x)

    def testRandomATA(self):
        np.random.seed(0)
        n = 5
        A0 = np.random.randn(n,n)
        A0 = np.eye(n) + A0.T.dot(A0)
        b0 = np.random.randn(n,1)
        x_init0 = np.zeros((n,1))
        self.verify(A0, b0, x_init0)

    def testRandomATAInitSolution(self):
        np.random.seed(0)
        n = 5
        A0 = np.random.randn(n,n)
        A0 = np.eye(n) + A0.T.dot(A0)
        x0 = np.random.randn(n,1)
        b0 = A0.dot(x0)
        x_init0 = np.zeros((n,1))
        self.verify(A0, b0, x0)


class ConjugateGradientLeastSquaresTest(tf.test.TestCase):
    @property
    def dtypes_to_test(self):
        return [tf.float32, tf.float64]

    def verify(self, A0, b0, shift0, x_init0):
        M = A0.T.dot(A0) + shift0*np.eye(A0.shape[1])
        expected_x = np.linalg.solve(M, A0.T.dot(b0))
        tol = 1e-6*np.linalg.norm(M.dot(x_init0) - A0.T.dot(b0))

        for dtype in self.dtypes_to_test:
            with self.test_session() as sess:
                A1 = tf.convert_to_tensor(A0, dtype=dtype)
                def A(x):
                    return tf.matmul(A1, x)
                def AT(x):
                    return tf.matmul(A1, x, transpose_a=True)

                b = tf.convert_to_tensor(b0, dtype=dtype)
                x_init = tf.convert_to_tensor(x_init0, dtype=dtype)
                shift = tf.convert_to_tensor(shift0, dtype=dtype)

                cgls = conjugate_gradient.ConjugateGradientLeastSquares(
                    A, AT, b, x_init, shift=shift, tol=1e-7)
                state = cgls.solve()
                self.assertEqual(dtype, state.x.dtype)

                state_np = sess.run(state)
                self.assertLess(state_np.k, cgls.max_iterations)
                self.assertAllClose(expected_x, state_np.x)

    def testRandomA(self):
        np.random.seed(0)
        m = 10
        n = 5
        A = np.random.randn(m,n)
        b = np.random.randn(m,1)
        x_init = np.zeros((n,1))
        shift = 1
        self.verify(A, b, shift, x_init)

    def testShiftZero(self):
        np.random.seed(0)
        m = 10
        n = 5
        A = np.random.randn(m,n)
        b = np.random.randn(m,1)
        x_init = np.zeros((n,1))
        shift = 0
        self.verify(A, b, shift, x_init)

    def testInitSolution(self):
        np.random.seed(0)
        m = 10
        n = 5
        A = np.random.randn(m,n)
        x = np.random.randn(n,1)
        b = A.dot(x)
        shift = 0
        self.verify(A, b, shift, x)


if __name__ == "__main__":
    tf.test.main()
