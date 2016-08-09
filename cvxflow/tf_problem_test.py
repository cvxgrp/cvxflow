
from nose.tools import assert_items_equal
import cvxpy as cvx
import numpy as np
import tensorflow as tf

from cvxflow import tf_problem

def assert_tensor_dict(sess, x, expected_values):
    values = dict(zip(x.values.keys(), sess.run(x.values.values())))
    assert_items_equal(values.keys(), expected_values.keys())
    for k, v in values.items():
        np.testing.assert_almost_equal(v, expected_values[k])


def test_tensor_problem():
    np.random.seed(0)
    m = 5
    n = 10
    A = np.abs(np.random.randn(m,n))
    b = A.dot(np.abs(np.random.randn(n)))
    c = np.random.rand(n)
    x = cvx.Variable(n)
    cvx_problem = cvx.Problem(cvx.Minimize(c.T*x), [A*x == b, x >= 0])
    obj, constrs = cvx_problem.canonicalize()

    # compare with problem data for SCS
    data = cvx_problem.get_problem_data(cvx.SCS)
    b0 = {constrs[0].constr_id: -data["b"][:m].reshape(-1,1),
          constrs[1].constr_id: -data["b"][m:m+n].reshape(-1,1)}
    c0 = {x.id: data["c"].reshape(-1,1)}

    x0 = np.random.randn(n,1)
    y0_0 = np.random.randn(m,1)
    y0_1 = np.random.randn(n,1)

    A0 = data["A"][:m,:]
    A1 = data["A"][m:m+n,:]
    Ax0 = {constrs[0].constr_id: A0*x0,
           constrs[1].constr_id: A1*x0}
    ATy0 = {x.id: A0.T*y0_0 + A1.T*y0_1}

    x_t = tf_problem.TensorDict({x.id: tf.constant(x0)})
    y_t = tf_problem.TensorDict({constrs[0].constr_id: tf.constant(y0_0),
                                 constrs[1].constr_id: tf.constant(y0_1)})

    problem = tf_problem.TensorProblem(cvx_problem)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        assert_tensor_dict(sess, problem.b, b0)
        assert_tensor_dict(sess, problem.c, c0)
        assert_tensor_dict(sess, problem.A(x_t), Ax0)
        assert_tensor_dict(sess, problem.AT(y_t), ATy0)
