
from numpy.testing import assert_allclose

from cvxflow.prox.functions import *
from cvxflow.prox.solvers import *

SOLVER_TESTS = [
    (ADMM(LeastSquares([[1.,2.],[3.,4.],[5.,6.]], [7.,8.,9.]),
          AbsoluteValue(1),
          variable_shape=2),
     ([-1.83334283, 3.20834076], None, None))
]

def get_solver_test(solver, expected_vars):
    def test(self):
        variables = solver.solve(verbose=False)
        for expected_var, var in zip(expected_vars, variables):
            if expected_var is None:
                continue
            assert_allclose(expected_var, var, atol=1e-4)
    return test

class SolverTest(tf.test.TestCase):
    pass

if __name__ == "__main__":
    counts = {}

    for params in SOLVER_TESTS:
        name = params[0].__class__.__name__
        count = counts.setdefault(name, 0)
        test_name = "test_" + name + str(count)
        setattr(SolverTest, test_name, get_solver_test(*params))
        counts[name] += 1

    tf.test.main()
