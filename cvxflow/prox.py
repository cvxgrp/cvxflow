import tensorflow as tf

from cvxflow import block_ops
from cvxflow import utils

class ProxFunction(object):
    """Abstract base class for proximal functions."""
    def __init__(self, name=None):
        self.name = name or type(self).__name__
        with tf.name_scope(self.name):
            with tf.name_scope("initialize"):
                self.initialize()

    def call(self, v):
        raise NotImplementedError

    def initialize(self):
        pass

    def __call__(self, v):
        with tf.name_scope(self.name):
            with tf.name_scope("call"):
                return self.call(v)

class AbsoluteValue(ProxFunction):
    """Absolute value.

    If lam is a scalar, lam*|x|
    If lam is a tuple, -lam[0]*neg(x) + lam[1]*pos(x)."""
    def __init__(self, scale=None, **kwargs):
        self.scale = utils.normalize_tuple(scale, 2, "scale")
        super(AbsoluteValue, self).__init__(**kwargs)

    def call(self, v):
        return (tf.maximum(v - self.scale[1], 0) +
                tf.minimum(v + self.scale[0], 0))

class BlockSeparable(ProxFunction):
    """Block separable prox functions."""
    def __init__(self, proxs=None, shapes=None, **kwargs):
        self.slices = block_ops.get_slices(shapes)
        self.proxs = proxs
        super(BlockSeparable, self).__init__(**kwargs)

    def call(self, vs):
        return block_ops.to_vector(
            [prox(v) for prox, v
             in zip(self.proxs, block_ops.to_list(vs, self.slices))])

class Nonnegative(ProxFunction):
    def call(self, v):
        return tf.maximum(v, 0)

class LeastSquares(ProxFunction):
    """Least squares.

    (1/2)||Ax - b||^2 + (mu/2)||x||^2 + I(Cx = d) + (1/2)||x - v||^2."""

    def __init__(self, A=None, b=None, C=None, d=None, mu=None, n=None,
                 dtype=tf.float32, **kwargs):
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.mu = mu if mu is not None else 0
        self.n = n
        self.dtype = dtype
        self.constrained = C is not None
        super(LeastSquares, self).__init__(**kwargs)

    def initialize(self):
        self.h = 0
        M = (self.mu+1)*tf.eye(self.n, dtype=self.dtype)
        if self.A is not None:
            self.A = tf.convert_to_tensor(self.A, dtype=self.dtype)
            M += tf.matmul(self.A, self.A, transpose_a=True)
            if self.b is not None:
                self.b = tf.convert_to_tensor(self.b, dtype=self.dtype)
                self.h = tf.matmul(self.A, self.b, transpose_a=True)
        self.chol = tf.cholesky(M)

        if self.constrained:
            self.C = tf.convert_to_tensor(self.C, dtype=self.dtype)
            if self.d is not None:
                self.d = tf.convert_to_tensor(self.d, dtype=self.dtype)

            # CM^{-1}C^T
            self.chol_constraint = tf.matmul(
                self.C,
                tf.cholesky_solve(self.chol, tf.transpose(self.C)))

    def call(self, v):
        if self.constrained:
            z = tf.cholesky_solve(self.chol, self.h + v)
            y = tf.cholesky_solve(
                self.chol_constraint, tf.matmul(self.C, z) - self.d)
            return tf.cholesky_solve(
                self.chol, self.h + v - tf.matmul(self.C, y, transpose_a=True))

        else:
            return tf.cholesky_solve(self.chol, self.h + v)

class Composition(ProxFunction):
    """Composition.

    f(Qx + b)."""
    def __init__(self, prox=None, Q=None, b=None, **kwargs):
        self.prox = prox
        self.Q = Q
        self.b = b
        super(Composition, self).__init__(**kwargs)

    def call(self, v):
        if self.Q is not None:
            v = tf.matmul(Q, v)
        if self.b is not None:
            v = v + self.b
        x = self.prox(v)
        if self.b is not None:
            x = x - self.b
        if self.Q is not None:
            x = tf.matmul(Q, x, transpose_a=True)
        return x
