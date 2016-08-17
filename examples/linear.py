"""Examples of solving linear systems with TensorFlow."""

import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from cvxflow import conjugate_gradient
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
SIZES = [100]


mu = 1e-4
for op in OPERATORS:
    for n in SIZES:
        print op.__name__, n

        np.random.seed(0)
        expr, x_var = op(n)
        b0 = expr.value

        # solve using spsolve
        prob = cvx.Problem(cvx.Minimize(0), [expr == 0])
        A0 = prob.get_problem_data(cvx.SCS)["A"]
        x1 = sp.linalg.spsolve(A0.T*A0 + mu*sp.eye(n), A0.T*b0).reshape(-1,1)
        print "spsolve norm:", np.linalg.norm(b0 - A0*x1)

        # solve using tensorflow
        expr = expr.canonicalize()[0]
        def A(x):
            return cvxpy_expr.tensor(expr, {x_var.id: x})
        def AT(y):
            return cvxpy_expr.adjoint_tensor(expr, y)[x_var.id]
        def M(x):
            return mu*x + AT(A(x))

        b = tf.constant(b0.reshape(-1,1), dtype=tf.float32)
        x_init = tf.zeros((n,1))
        x = conjugate_gradient.solve(M, AT(b), x_init)

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            x2 = sess.run(x)
        print "tensorflow norm:", np.linalg.norm(b0 - A0*x2)
