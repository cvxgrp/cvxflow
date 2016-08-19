"""Examples of solving linear systems with TensorFlow."""

import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import time

from cvxflow.conjugate_gradient import *
from cvxflow import cvxpy_expr

def dense_matrix(n):
    x = cvx.Variable(n)
    A = np.random.randn(2*n, n)
    x.value = np.random.randn(n)
    return A*x, x

def sparse_matrix(n):
    x = cvx.Variable(n)
    A = sp.rand(2*n, n, 0.01)
    A.data = np.random.randn(A.nnz)
    x.value = np.random.randn(n)
    return A*x, x

def convolution(n):
    x = cvx.Variable(n)
    sigma = n/10
    c = np.exp(-np.arange(-n/2., n/2.)**2./(2*sigma**2))/np.sqrt(2*sigma**2*np.pi)
    c[c < 1e-4] = 0

    k = 5
    x.value = np.zeros(n)
    x.value[np.random.choice(n,k),0] = np.random.rand(k)*sigma
    return cvx.conv(c, x), x

OPERATORS = [
    dense_matrix,
    sparse_matrix,
    convolution
]
SIZES = [1000]


def nnz(Ax_expr):
    prob = cvx.Problem(cvx.Minimize(0), [Ax_expr == 0])
    return prob.get_problem_data(cvx.SCS)["A"].nnz

lam = 1e-4
sigma = 1e-2
for op in OPERATORS:
    for n in SIZES:
        print op.__name__
        print "num_vars:", n

        np.random.seed(0)
        Ax_expr, x_var = op(n)
        b0 = Ax_expr.value
        b0 += sigma*np.random.randn(*b0.shape)
        obj = cvx.sum_squares(Ax_expr - b0) + lam*cvx.sum_squares(x_var)
        prob = cvx.Problem(cvx.Minimize(obj))
        print "nnz:", nnz(Ax_expr)

        print "spsolve"
        t0 = time.time()
        prob.solve(solver=cvx.LS)
        print "solve_time: %.2f secs" % (time.time() - t0)
        print "objective: %.3e" % obj.value

        print "tensorflow"
        t0 = time.time()
        expr = Ax_expr.canonicalize()[0]
        def A(x):
            return cvxpy_expr.tensor(expr, {x_var.id: x})
        def AT(y):
            return cvxpy_expr.adjoint_tensor(expr, y)[x_var.id]
        def M(x):
            return lam*x + AT(A(x))

        b = tf.constant(b0.reshape(-1,1), dtype=tf.float32)
        x_init = tf.zeros((n,1))
        x, iters, r_norm_sq = conjugate_gradient_solve(M, AT(b), x_init)

        init = tf.initialize_all_variables()
        print "init_time: %.2f secs" % (time.time() - t0)

        t0 = time.time()
        with tf.Session() as sess:
            sess.run(init)
            x0, iters0, r_norm_sq0 = sess.run([x, iters, r_norm_sq])
        print "cpu_solve_time: %.2f secs" % (time.time() - t0)

        x_var.value = x0
        print "objective: %.3e" % obj.value
        print "cg_iterations:", iters0
