


import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from cvxflow.equilibrate import equilibrate

def get_alpha_beta(m, n):
    return (n / m)**(0.25), (m / n)**(0.25)


def project(x, M):
    """Project x onto [-M, M]^n.
    """
    return np.minimum(M, np.maximum(x, -M, out=x), out=x)

# Comparison method.


def f(A, u, v, gamma, p=2):
    m, n = A.shape
    alpha, beta = get_alpha_beta(m, n)
    total = (1. / p) * np.exp(p * u).T.dot(np.power(np.abs(A), p)).dot(np.exp(p * v))
    total += -alpha**p * u.sum() - beta**p * v.sum() + (gamma / 2) * ((u * u).sum() + (v * v).sum())
    return np.sum(total)


def get_grad(A, u, v, gamma, p=2):
    m, n = A.shape
    alpha, beta = get_alpha_beta(m, n)

    tmp = np.diag(np.exp(p * u)).dot((A * A)).dot(np.exp(p * v))
    grad_u = tmp - alpha**p + gamma * u
    du = -grad_u / (2 * tmp + gamma)

    tmp = np.diag(np.exp(p * v)).dot((A.T * A.T)).dot(np.exp(p * u))
    grad_v = tmp - beta**p + gamma * v
    dv = -grad_v / (2 * tmp + gamma)

    return du, dv, grad_u, grad_v


def newton_equil(A, gamma, max_iters):
    alpha = 0.25
    beta = 0.5
    m, n = A.shape
    u = np.zeros(m)
    v = np.zeros(n)
    for i in range(max_iters):
        du, dv, grad_u, grad_v = get_grad(A, u, v, gamma)
        # Backtracking line search.
        t = 1
        obj = f(A, u, v, gamma)
        grad_term = np.sum(alpha * (grad_u.dot(du) + grad_v.dot(dv)))
        while True:
            new_obj = f(A, u + t * du, v + t * dv, gamma)
            if new_obj > obj + t * grad_term:
                t = beta * t
            else:
                u = u + t * du
                v = v + t * dv
                break
    return np.exp(u), np.exp(v)

class EquilibrateTest(tf.test.TestCase):
    def test_eye(self):
        np.random.seed(0)
        tf.set_random_seed(0)
        n = 5
        m = 5
        A0 = np.eye(n)
        _A = tf.constant(A0, dtype=tf.float32)
        def A(x):
            return tf.matmul(_A, x)
        u0 = np.ones((n, 1))
        v0 = np.ones((m, 1))
        u, v = equilibrate(A, A, (m, n), 10, gamma=1e-1)
        init = tf.initialize_all_variables()
        with self.test_session():
            tf.initialize_all_variables().run()
            uval = u.eval()
            vval = v.eval()
            assert_allclose(uval*vval, np.ones((n, 1)), rtol=0, atol=1e-6)

    def test_small(self):
        np.random.seed(0)
        tf.set_random_seed(0)
        n = 2
        m = 3
        A0 = np.ones((m, n))
        u0 = np.arange(m) + 1
        v0 = np.arange(n) + 1
        A0 = np.diag(u0.flatten()).dot(A0).dot(np.diag(v0.flatten()))
        _A = tf.constant(A0, dtype=tf.float32)
        def A(x):
            return tf.matmul(_A, x)
        _AT = tf.constant(A0.T, dtype=tf.float32)
        def AT(x):
            return tf.matmul(_AT, x)
        gamma = 1e-1
        # compare with reference solution.
        d_newt, e_newt = newton_equil(A0, gamma, 100)
        sltn_val = f(A0, np.log(d_newt), np.log(e_newt), gamma)

        u, v = equilibrate(A, AT, (m, n), 250, gamma=gamma, M=3.)
        with self.test_session() as sess:
            d, e = sess.run([u, v])
            d = d.ravel()
            e = e.ravel()
            obj_val = f(A0, np.log(d), np.log(e), gamma)
            assert abs(obj_val - sltn_val) <= 5

if __name__ == "__main__":
    tf.test.main()
