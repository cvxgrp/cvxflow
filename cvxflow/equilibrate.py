from __future__ import division
"""Matrix-free equilibration."""

import tensorflow as tf
import math

def balance(A, AT, shape, iters=50, name=None, seed=None):
    """Scales rows once and columns once (with stochastic estimates).
    """
    m, n = shape
    with tf.op_scope([A, AT], name, "balance"):
        if iters == 0:
            return tf.ones((m, 1)), tf.ones((n, 1))
        else:
            alpha, beta = get_alpha_beta(m, n)
            d = tf.zeros((m, 1))
            for i in range(iters):
                s = random_probe((n, 1))
                d += tf.square(A(s))
            tmp = tf.sqrt(d/iters)
            d = alpha*tf.inv(project(tmp, 1e-3, 1e3))
            e = tf.zeros((n, 1))
            for i in range(iters):
                w = random_probe((m, 1))
                e += tf.square(AT(d * w))
            tmp = tf.sqrt(e/iters)
            e = beta*tf.inv(project(tmp, 1e-3, 1e3))
            return d, e

def equilibrate(A, AT, shape, iters=50, gamma=1e-1, M=math.log(1e3),
                name=None, seed=None):
    """Computes diagonal D, E so that DAE is approximately equilibrated.
    """
    with tf.op_scope([A, AT], name, "equilibrate"):
        m, n = shape
        alpha, beta = get_alpha_beta(m, n)

        u = tf.zeros((m, 1))
        v = tf.zeros((n, 1))
        ubar = tf.zeros((m, 1))
        vbar = tf.zeros((n, 1))

        # Main loop.
        for t in range(1, iters + 1):
            step_size = 2 / (gamma * (t + 1))
            # u grad estimate.
            s = random_probe((n, 1), seed)
            As = A(tf.exp(v) * s)
            u_grad = tf.exp(2 * u) * tf.square(As) - alpha**2 + gamma * u

            # v grad estimate.
            w = random_probe((m, 1), seed)
            ATu = AT(tf.exp(u) * w)
            v_grad = tf.exp(2 * v) * tf.square(ATu) - beta**2 + gamma * v

            u = project(u - step_size * u_grad, -M, M)
            v = project(v - step_size * v_grad, -M, M)
            # Update averages.
            ubar = 2 * u / (t + 2) + t * ubar / (t + 2)
            vbar = 2 * v / (t + 2) + t * vbar / (t + 2)

        return tf.exp(ubar), tf.exp(vbar)


def random_probe(shape, seed=None):
    """Random +1/-1 entries."""
    samples = tf.to_float(tf.random_uniform(shape, seed=seed) > 0.5)
    return 2*samples-1


def get_alpha_beta(m, n):
    return (n / m)**(0.25), (m / n)**(0.25)


def project(x, L, U):
    """Project x onto [L,U]^n.
    """
    return tf.minimum(U, tf.maximum(x, L))
