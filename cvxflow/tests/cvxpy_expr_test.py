
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
    ("conv", cvx.conv([1,2,3], x_var), [1,2,4], [1,2,4,6,8]),
    ("sparse", A_sparse*x_var, [1,2,4], [1,2,4,6,8]),
]

class TensorTest(tf.test.TestCase):
    pass

def get_tensor_test(f_expr, x, y):
    f = f_expr.canonicalize()[0]
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)

    def test(self):
        prob = cvx.Problem(cvx.Minimize(0), [f_expr == 0])
        A = prob.get_problem_data(cvx.SCS)["A"]
        xt = tf.constant(x, dtype=tf.float32)
        yt = tf.constant(y, dtype=tf.float32)

        Ax = cvxpy_expr.tensor(f, {x_var.id: xt})
        ATy = cvxpy_expr.adjoint_tensor(f, yt)[x_var.id]

        with self.test_session():
            assert_allclose(Ax.eval(), A*x)
            assert_allclose(ATy.eval(), A.T*y)

    return test

if __name__ == "__main__":
    for name, f, x, y in EXPRESSIONS:
        test_name = "test_" + name
        setattr(TensorTest, test_name, get_tensor_test(f, x, y))
    tf.test.main()
