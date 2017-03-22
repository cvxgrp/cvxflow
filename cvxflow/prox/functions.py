
import tensorflow as tf

class ProxFunction(object):
    def __repr__(self):
        return self.__class__.__name__

    def initialize_graph(self):
        pass


class AbsoluteValue(ProxFunction):
    """Absolute value.

    If lam is a scalar, lam*|x|
    If lam is a tuple, -lam[0]*neg(x) + lam[1]*pos(x)."""
    def __init__(self, lam):
        self.lam = lam
        if not isinstance(self.lam, tuple):
            self.lam = (lam, lam)

    def __call__(self, v):
        return (tf.maximum(v - self.lam[1], 0) +
                tf.minimum(v + self.lam[0], 0))


class LeastSquares(ProxFunction):
    """Least squares.

    0.5*sum_squares(A*x - b) + 0.5*mu*sum_squares(x)."""
    def __init__(self, A, b, mu=0):
        self.A = A
        self.b = b
        self.mu = mu

    def initialize_graph(self):
        self.A = tf.constant(self.A)
        self.b = tf.constant(self.b, shape=(len(self.b), 1))

        m, n = self.A.get_shape()
        if m > n:
            ATA = tf.matmul(self.A, self.A, transpose_a=True)
            self.L = tf.cholesky(ATA + (self.mu+1)*tf.eye(int(n)))
        else:
            # TODO(mwytock): Add fat case
            raise NotImplementedError

        self.ATb = tf.matmul(self.A, self.b, transpose_a=True)

    def __call__(self, v):
        v = tf.expand_dims(v, -1)
        return tf.squeeze(tf.cholesky_solve(self.L, self.ATb + v))
