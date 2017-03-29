"""Solve a linear system with conjugate gradient."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

from cvxflow import vector_ops

def cg_solve(A, b, x_init, tol=1e-6, return_info=False, max_iterations=100,
             name=None):
  """Solve linear system `Ax = b`, using conjugate gradient."""
  with ops.name_scope(name, "ConjugateGradientSolve", [b, x_init]):
    # initial values
    x = x_init
    r = b - A(x)
    p = r
    r_norm_sq = vector_ops.dot(r,r)
    k = constant_op.constant(0)
    init = (x, r, p, r_norm_sq, k)

    r_norm_sq0 = r_norm_sq
    def cond(x, r, p, r_norm_sq, k):
      return math_ops.logical_and(
        r_norm_sq > tol*tol*r_norm_sq0,
        k < max_iterations)

    def body(x, r, p, r_norm_sq, k):
      Ap = A(p)
      alpha = r_norm_sq / vector_ops.dot(p, Ap)
      x = x + alpha*p
      r = r - alpha*Ap
      r_norm_sq_prev = r_norm_sq
      r_norm_sq = vector_ops.dot(r,r)
      beta = r_norm_sq / r_norm_sq_prev
      p = r + beta*p
      return x, r, p, r_norm_sq, k+1

    result = control_flow_ops.while_loop(cond, body, init)
    if return_info:
      return result
    else:
      return result[0]

def cgls_solve(A, AT, b, x_init, shift=0, tol=1e-6, name=None,
               max_iterations=100, return_info=False):
  """Solve using conjugate gradient for least squares.

  Solves the linear system `(A'A + sI)x = b`."""
  with ops.name_scope(name, "ConjugateGradientLSSolve", [shift, b, x_init]):
    # initial values
    x = x_init
    r = b - A(x)
    p = AT(r) - shift*x
    s_norm_sq = vector_ops.dot(p,p)
    k = constant_op.constant(0)
    init = (x, r, p, s_norm_sq, k)

    s_norm_sq0 = s_norm_sq
    def cond(x, r, p, s_norm_sq, k):
      return math_ops.logical_and(
        s_norm_sq > tol*tol*s_norm_sq0,
        k < max_iterations)

    def body(x, r, p, s_norm_sq, k):
      q = A(p)
      alpha = s_norm_sq / (vector_ops.dot(q,q) + shift*vector_ops.dot(p,p))
      x = x + alpha*p
      r = r - alpha*q
      s = AT(r) - shift*x
      s_norm_sq_prev = s_norm_sq
      s_norm_sq = vector_ops.dot(s,s)
      beta = s_norm_sq / s_norm_sq_prev
      p = s + beta*p
      return x, r, p, s_norm_sq, k+1

    result = control_flow_ops.while_loop(cond, body, init)
    if return_info:
      return result
    else:
      return result[0]
