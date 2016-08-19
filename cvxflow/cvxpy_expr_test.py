
from numpy.testing import assert_allclose
import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from cvxflow import cvxpy_expr

np.random.seed(0)
A_sparse = sp.rand(5, 3, 0.5)
x_var = cvx.Variable(3)

EXPRESSIONS = [
    (cvx.conv([1,2,3], x_var), [1,2,4], [1,2,4,6,8]),
    (A_sparse*x_var, [1,2,4], [1,2,4,6,8]),
]

def run_tensor(f, x, y):
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)

    prob = cvx.Problem(cvx.Minimize(0), [f == 0])
    A = prob.get_problem_data(cvx.SCS)["A"]
    xt = tf.constant(x, dtype=tf.float32)
    yt = tf.constant(y, dtype=tf.float32)

    f = f.canonicalize()[0]
    Ax = cvxpy_expr.tensor(f, {x_var.id: xt})
    ATy = cvxpy_expr.adjoint_tensor(f, yt)[x_var.id]

    with tf.Session() as sess:
        assert_allclose(sess.run(Ax), A*x)
        assert_allclose(sess.run(ATy), A.T*y)

def test_tensor():
    for f, x, y in EXPRESSIONS:
        yield run_tensor, f, x, y
