"""Solve a linear system with conjugate gradient."""

from collections import namedtuple
import tensorflow as tf

from cvxflow import vector_ops
from cvxflow.solvers.iterative_solver import IterativeSolver


class ConjugateGradient(IterativeSolver):
    def __init__(self, A, b, x_init, tol=1e-6, max_iterations=100, **kwargs):
        self.A = A
        self.b = b
        self.x_init = x_init
        self.tol = tol
        self.max_iterations = max_iterations
        super(ConjugateGradient, self).__init__(**kwargs)

    @property
    def state(self):
        return namedtuple("State", ["x", "r", "p", "r_norm_sq", "k"])

    def init(self):
        x = self.x_init
        r = self.b - self.A(x)
        p = r
        r_norm_sq = vector_ops.dot(r,r)
        k = tf.constant(0)
        self.r_norm_sq0 = r_norm_sq
        return self.state(x, r, p, r_norm_sq, k)

    def iterate(self, state):
        Ap = self.A(state.p)
        alpha = state.r_norm_sq / vector_ops.dot(state.p, Ap)
        x = state.x + alpha*state.p
        r = state.r - alpha*Ap
        r_norm_sq = vector_ops.dot(r,r)
        beta = r_norm_sq / state.r_norm_sq
        p = r + beta*state.p
        return self.state(x, r, p, r_norm_sq, state.k+1)

    def stop(self, state):
        return tf.logical_or(
            state.k >= self.max_iterations,
            state.r_norm_sq <= self.tol*self.tol*self.r_norm_sq0)


class ConjugateGradientLeastSquares(IterativeSolver):
    def __init__(self, A, AT, b, x_init, shift=0, tol=1e-6, max_iterations=100,
                 **kwargs):
        self.A = A
        self.AT = AT
        self.b = b
        self.x_init = x_init
        self.shift = shift
        self.tol = tol
        self.max_iterations = max_iterations
        super(ConjugateGradientLeastSquares, self).__init__(**kwargs)

    @property
    def state(self):
        return namedtuple("State", ["x", "r", "p", "s_norm_sq", "k"])

    def init(self):
        x = self.x_init
        r = self.b - self.A(x)
        p = self.AT(r) - self.shift*x
        s_norm_sq = vector_ops.dot(p,p)
        k = tf.constant(0)
        self.s_norm_sq0 = s_norm_sq
        return self.state(x, r, p, s_norm_sq, k)

    def iterate(self, state):
        q = self.A(state.p)
        alpha = (state.s_norm_sq /
                 (vector_ops.dot(q,q) +
                  self.shift*vector_ops.dot(state.p,state.p)))
        x = state.x + alpha*state.p
        r = state.r - alpha*q
        s = self.AT(r) - self.shift*x
        s_norm_sq = vector_ops.dot(s,s)
        beta = s_norm_sq / state.s_norm_sq
        p = s + beta*state.p
        return self.state(x, r, p, s_norm_sq, state.k+1)

    def stop(self, state):
        return tf.logical_or(
            state.k >= self.max_iterations,
            state.s_norm_sq <= self.tol*self.tol*self.s_norm_sq0)
