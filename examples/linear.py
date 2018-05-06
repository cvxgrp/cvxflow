"""Examples of solving linear systems with TensorFlow."""

import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
import time
import sys

def op_dense_matrix(n):
    x = cvx.Variable(n)
    A = np.random.randn(2*n, n)
    A /= np.sqrt(np.sum(A**2, 0))
    x.value = np.random.randn(n)
    return A*x, x

def op_sparse_matrix(n):
    x = cvx.Variable(n)
    A = sp.rand(2*n, n, 0.01)
    A.data = np.random.randn(A.nnz)
    N = A.copy()
    N.data = N.data**2
    A = A*sp.diags([1 / np.sqrt(np.ravel(N.sum(axis=0)))], [0])

    x.value = np.random.randn(n)
    return A*x, x

def op_convolution(n):
    x = cvx.Variable(n)
    sigma = n/10
    c = np.exp(-np.arange(-n/2., n/2.)**2./(2*sigma**2))/np.sqrt(2*sigma**2*np.pi)
    c[c < 1e-4] = 0

    k = 5
    x.value = np.zeros(n)
    x.value[np.random.choice(n,k),0] = np.random.rand(k)*sigma
    return cvx.conv(c, x), x

lam = 1e-4
sigma = 1e-2

def run_spsolve(Ax_expr, x_var):
    b0 = Ax_expr.value
    b0 += sigma*np.random.randn(*b0.shape)
    obj = cvx.sum_squares(Ax_expr - b0) + lam*cvx.sum_squares(x_var)
    prob = cvx.Problem(cvx.Minimize(obj))

    t0 = time.time()
    prob.solve(solver=cvx.LS)
    print "solve_time: %.2f secs" % (time.time() - t0)
    print "objective: %.3e" % obj.value

def run_tensorflow(Ax_expr, x_var):
    import tensorflow as tf

    from cvxflow.solvers import conjugate_gradient
    from cvxflow import expressions

    b0 = Ax_expr.value
    b0 += sigma*np.random.randn(*b0.shape)
    obj = cvx.sum_squares(Ax_expr - b0) + lam*cvx.sum_squares(x_var)

    t0 = time.time()
    expr = Ax_expr.canonicalize()[0]
    def A(x):
        return expressions.tensor(expr, {x_var.id: x})
    def AT(y):
        return expressions.adjoint_tensor(expr, y)[x_var.id]
    def M(x):
        return lam*x + AT(A(x))

    b = tf.constant(b0.reshape(-1,1), dtype=tf.float32)
    x_init = tf.zeros((n,1))
    solver = conjugate_gradient.ConjugateGradient(M, AT(b), x_init)
    state = solver.solve() 

    init = tf.global_variables_initializer()
    print "init_time: %.2f secs" % (time.time() - t0)

    t0 = time.time()
    with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
        sess.run(init)
        state_np = sess.run(state)
        print "cpu_solve_time: %.2f secs" % (time.time() - t0)

    t0 = time.time()
    with tf.Session() as sess:
        sess.run(init)
        state_np = sess.run(state)
        print "gpu_solve_time: %.2f secs" % (time.time() - t0)

    x_var.value = state_np.x
    print "objective: %.3e" % obj.value
    print "cg_iterations:", state_np.k


if __name__ == "__main__":
    _, run_name, op_name = sys.argv
    run = globals()["run_" + run_name]
    op = globals()["op_" + op_name]
    n = 1000

    print "running", op_name, run_name, n
    np.random.seed(0)
    run(*op(n))
    print
    print
