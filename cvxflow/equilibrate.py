
"""Matrix-free equilibration."""

import tensorflow as tf
import math


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
            s = random_probe(n, seed)
            As = A(tf.exp(v) * s)
            u_grad = tf.exp(2 * u) * tf.square(As) - alpha**2 + gamma * u

            # v grad estimate.
            w = random_probe(m, seed)
            ATu = AT(tf.exp(u) * w)
            v_grad = tf.exp(2 * v) * tf.square(ATu) - beta**2 + gamma * v

            u = project(u - step_size * u_grad, M)
            v = project(v - step_size * v_grad, M)
            # Update averages.
            ubar = 2 * u / (t + 2) + t * ubar / (t + 2)
            vbar = 2 * v / (t + 2) + t * vbar / (t + 2)

        return tf.exp(ubar), tf.exp(vbar)


def random_probe(dim, seed):
    """Random +1/-1 entries."""
    samples = tf.to_float(tf.random_uniform((dim, 1), seed=seed) > 0.5)
    return 2*samples-1


def get_alpha_beta(m, n):
    return (n / m)**(0.25), (m / n)**(0.25)


def project(x, M):
    """Project x onto [-M, M]^n.
    """
    return tf.minimum(M, tf.maximum(x, -M))
