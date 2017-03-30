
import numpy as np
import tensorflow as tf

from cvxflow import block_ops
from cvxflow import prox
from cvxflow.solvers import pogs

class POGSTest(tf.test.TestCase):
    @property
    def dtypes_to_test(self):
        return [tf.float32, tf.float64]


class LassoTest(POGSTest):
    def verify(self, A0, b0, lam0, expected_x):
        for dtype in self.dtypes_to_test:
            with self.test_session() as sess:
                A1 = tf.convert_to_tensor(A0, dtype=dtype)
                m, n = A1.get_shape().as_list()
                b = tf.convert_to_tensor(b0, dtype=dtype)
                lam = tf.convert_to_tensor(lam0, dtype=dtype)
                I = tf.eye(m, dtype=dtype)

                def A(x):
                    return tf.matmul(A1, x)
                def AT(y):
                    return tf.matmul(A1, y, transpose_a=True)

                solver = pogs.POGS(
                    prox_f=prox.LeastSquares(A=I, b=b, n=m, dtype=dtype),
                    prox_g=prox.AbsoluteValue(scale=lam),
                    A=A,
                    AT=AT,
                    shape=(m, n),
                    dtype=dtype)

                state = solver.solve()
                self.assertEqual(dtype, state.x.dtype)

                state_np = sess.run(state)
                self.assertLess(state_np.k, solver.max_iterations)
                self.assertAllClose(expected_x, state_np.x, rtol=1e-2, atol=1e-4)

    def testLasso(self):
        self.verify([[1.,-10],[1.,10.],[1.,0.]], [[2.],[2.],[2.]], 1,
                    [[1.6666666], [0]])


class OrthogonalLassoTest(POGSTest):
    def verify(self, A0, b, lam, expected_x):
        U0, s0, VT0 = np.linalg.svd(A0, full_matrices=False)
        m, n = U0.shape

        for dtype in self.dtypes_to_test:
            with self.test_session() as sess:
                U = tf.convert_to_tensor(U0, dtype=dtype)
                V = tf.convert_to_tensor(VT0.T, dtype=dtype)
                s = tf.convert_to_tensor(s0.reshape(-1,1), dtype=dtype)

                I = tf.eye(m, dtype=dtype)
                prox_f = prox.LeastSquares(A=I, b=b, n=m, dtype=dtype)
                prox_g = prox.AbsoluteValue(scale=lam)
                def prox_f_tilde(v):
                    return tf.matmul(U, prox_f(tf.matmul(U, v)), transpose_a=True)
                def prox_g_tilde(v):
                    return tf.matmul(V, prox_g(tf.matmul(V, v)), transpose_a=True)
                def project_linear_diag(x, y):
                    x = (x + s*y)/(1 + s*s)
                    y = s*x
                    return x, y

                solver = pogs.POGS(
                    prox_f=prox_f_tilde,
                    prox_g=prox_g_tilde,
                    A=lambda x: s*x,
                    project_linear=project_linear_diag,
                    shape=(n, n),
                    dtype=dtype)

                state = solver.solve()
                x = tf.matmul(V, state.x)
                self.assertEqual(dtype, x.dtype)
                self.assertLess(sess.run(state.k), solver.max_iterations)
                self.assertAllClose(expected_x, sess.run(x), rtol=1e-2, atol=1e-4)

    def testOrthogonalLasso(self):
        self.verify([[1.,-10],[1.,10.],[1.,0.]], [[2.],[2.],[2.]], 1,
                    [[1.6666666], [0]])


class MultipleQuantileRegressionTest(POGSTest):
    def verify(self, X0, y0, expected_obj_val):
        for dtype in self.dtypes_to_test:
            with self.test_session() as sess:
                tau = np.array([0.2,0.5,0.8])
                X = tf.convert_to_tensor(X0, dtype=dtype)
                y = tf.convert_to_tensor(y0, dtype=dtype)
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
                    XT = tf.matmul(X, theta)
                    XTD = XT[:,1:] - XT[:,:-1]
                    return block_ops.to_vector([XT, XTD])
                def AT(y):
                    W, Z = block_ops.to_list(y, y_slices)
                    XTW = tf.matmul(X, W, transpose_a=True)
                    XTZ = tf.matmul(X, Z, transpose_a=True)
                    XTZDT = tf.concat(
                        [-XTZ[:,:1], XTZ[:,:-1] - XTZ[:,1:], XTZ[:,-1:]], axis=1)
                    return block_ops.to_vector([XTW + XTZDT])

                scale = (tf.convert_to_tensor(tau, dtype=dtype),
                         tf.convert_to_tensor(1-tau, dtype=dtype))
                tilted_l1 = prox.AbsoluteValue(scale=scale)
                def prox_quantile_loss(v):
                    return tilted_l1(v-y) + y

                solver = pogs.POGS(
                    prox_f=prox.BlockSeparable(
                        proxs=[prox_quantile_loss, prox.Nonnegative()],
                        shapes=y_shape),
                    prox_g=prox.LeastSquares(mu=1e-2, n=n*k, dtype=dtype),
                    A=A,
                    AT=AT,
                    shape=(m*k+m*(k-1), n*k),
                    dtype=dtype,
                    max_iterations=3000)

                state = solver.solve()
                self.assertEqual(dtype, state.x.dtype)

                state_np = sess.run(state)
                self.assertLess(state_np.k, solver.max_iterations)

                theta = state_np.x.reshape(n,k)
                z = np.array(X0).dot(theta) - np.array(y0)
                obj_val = (np.sum(-tau*np.minimum(z, 0) + (1-tau)*np.maximum(z, 0)) +
                           1e-2/2*np.sum(theta**2))
                self.assertAllClose(expected_obj_val, obj_val, rtol=1e-2, atol=1e-4)

    def testMultipleQuantileRegression(self):
        self.verify([[1.,1.],[1.,2.],[1.,3.],[1.,4.]], [[1.],[2.],[10.],[20.]],
                     10.0051362724)


if __name__ == "__main__":
    tf.test.main()
