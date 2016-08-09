

import numpy as np
import tensorflow as tf

from cvxflow import conjugate_gradient

def test_solve():
    np.random.seed(0)
    n = 5
    A0 = np.random.randn(n, n)
    A0 = np.eye(n) + A0.T.dot(A0)
    _A = tf.constant(A0)
    def A(x):
        return tf.matmul(_A, x)

    b0 = np.random.randn(n, 1)
    b = tf.constant(b0)
    x = conjugate_gradient.solve(A, b, tf.zeros((n,1), dtype=tf.float64))

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        np.testing.assert_allclose(sess.run(x), np.linalg.solve(A0, b0))
