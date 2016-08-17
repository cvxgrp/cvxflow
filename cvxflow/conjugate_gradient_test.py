

import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from cvxflow.conjugate_gradient import *


def test_solve():
    np.random.seed(0)
    n = 5
    A0 = np.random.randn(n, n)
    A0 = np.eye(n) + A0.T.dot(A0)
    _A = tf.constant(A0, dtype=tf.float32)
    def A(x):
        return tf.matmul(_A, x)

    b0 = np.random.randn(n, 1)
    x0 = np.linalg.solve(A0, b0)

    b = tf.constant(b0, dtype=tf.float32)
    x, _, _ = conjugate_gradient_solve(A, b, tf.zeros((n,1), dtype=tf.float32))
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        assert_allclose(sess.run(x), x0, rtol=0, atol=1e-6)
