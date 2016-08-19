
import cvxpy as cvx
import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from cvxflow.problem import TensorProblem
from cvxflow.problem_testutil import PROBLEMS

class ProblemTest(tf.test.TestCase):
    pass

def get_test(problem_gen):
    def test(self):
        np.random.seed(0)
        cvx_problem = problem_gen()
        data = cvx_problem.get_problem_data(cvx.SCS)
        m, n = data["A"].shape
        x0 = np.random.randn(n,1)
        y0 = np.random.randn(m,1)
        x = tf.constant(x0, dtype=tf.float32)
        y = tf.constant(y0, dtype=tf.float32)

        problem = TensorProblem(cvx_problem)
        with self.test_session():
            tf.initialize_all_variables().run()
            assert_allclose(problem.b.eval(), data["b"].reshape(-1,1), rtol=0, atol=1e-6)
            assert_allclose(problem.c.eval(), data["c"].reshape(-1,1), rtol=0, atol=1e-6)
            assert_allclose(problem.A(x).eval(), data["A"]*x0, rtol=0, atol=1e-6)
            assert_allclose(problem.AT(y).eval(), data["A"].T*y0, rtol=0, atol=1e-6)
    return test


if __name__ == "__main__":
    for problem in PROBLEMS:
        test_name = "test_%s" % problem.__name__
        setattr(ProblemTest, test_name, get_test(problem))
    tf.test.main()
