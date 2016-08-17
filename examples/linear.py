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

def convolution(n):
    x = cvx.Variable(n)
    sigma = n/10
    k = 5
    c = np.exp(-np.arange(-n/2., n/2.)**2./(2*sigma**2))/np.sqrt(2*sigma**2*np.pi)
    x.value = np.zeros(n)
    x.value[np.random.choice(n,k),0] = np.random.rand(k)*sigma
    return cvx.conv(c, x), x

OPERATORS = [
    dense_matrix,
    convolution
]
SIZES = [100, 1000]


mu = 1e-4
for op in OPERATORS:
    for n in SIZES:
        print op.__name__, n

        np.random.seed(0)
        expr, x_var = op(n)
        b0 = expr.value

        # solve using spsolve
        t0 = time.time()
        prob = cvx.Problem(cvx.Minimize(0), [expr == 0])
        A0 = prob.get_problem_data(cvx.SCS)["A"]
        x1 = sp.linalg.spsolve(A0.T*A0 + mu*sp.eye(n), A0.T*b0).reshape(-1,1)
        print "spsolve:", np.linalg.norm(b0 - A0*x1), time.time() - t0

        # solve using tensorflow
        t0 = time.time()
        expr = expr.canonicalize()[0]
        def A(x):
            return cvxpy_expr.tensor(expr, {x_var.id: x})
        def AT(y):
            return cvxpy_expr.adjoint_tensor(expr, y)[x_var.id]
        def M(x):
            return mu*x + AT(A(x))

        b = tf.constant(b0.reshape(-1,1), dtype=tf.float32)
        x_init = tf.zeros((n,1))
        x, iters, r_norm_sq = conjugate_gradient_solve(M, AT(b), x_init)

        init = tf.initialize_all_variables()
        print "tensorflow_init:", time.time() - t0

        t0 = time.time()
        with tf.Session() as sess:
            sess.run(init)
            x2, iters0, r_norm_sq0 = sess.run([x, iters, r_norm_sq])
        print "tensorflow:", np.linalg.norm(b0 - A0*x2), iters0, r_norm_sq0, time.time() - t0
