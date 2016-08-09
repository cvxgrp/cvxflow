"""Solve a linear system with conjugate gradient."""

import tensorflow as tf

from cvxflow.tf_util import dot

def solve(A, b, x, tol=1e-10):
    def body(x, r, p, r_norm_sq):
        Ap = A(p)
        alpha = r_norm_sq / dot(p, Ap)
        x = x + alpha*p
        r = r - alpha*Ap
        r_norm_sq_prev = r_norm_sq
        r_norm_sq = dot(r, r)
        beta = r_norm_sq / r_norm_sq_prev
        p = r + beta*p
        return (x, r, p, r_norm_sq)

    def cond(x, r, p, r_norm_sq):
        return tf.sqrt(r_norm_sq) > tol

    r = b - A(x)
    loop_vars = (x, r, r, dot(r, r))
    return tf.while_loop(cond, body, loop_vars)[0]
