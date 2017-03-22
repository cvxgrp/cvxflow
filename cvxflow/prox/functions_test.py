
import tensorflow as tf
from numpy.testing import assert_allclose

from cvxflow.prox.functions import *

PROX_TESTS = [
    (AbsoluteValue(2), [-3,-1,4], [-1,0,2]),
    (AbsoluteValue((3, 1)), [-4,-1,3], [-1,0,2]),
    (LeastSquares([[1.,2.],[3.,4.],[5.,6.]], [7.,8.,9.]), [1.,2.], [-0.853448, 2.448276]),
    (LeastSquares([[1.,2.],[3.,4.],[5.,6.]], [7.,8.,9.], 0.1), [1.,2.], [-0.728593, 2.347778]),
]

def get_prox_test(f, v, expected_x):
    def test(self):
        with self.test_session():
            f.initialize_graph()
            x = f(tf.constant(v))
            assert_allclose(expected_x, x.eval(), atol=1e-5)

    return test

class ProxTest(tf.test.TestCase):
    pass

if __name__ == "__main__":
    counts = {}

    for params in PROX_TESTS:
        name = params[0].__class__.__name__
        count = counts.setdefault(name, 0)
        test_name = "test_" + name + str(count)
        setattr(ProxTest, test_name, get_prox_test(*params))
        counts[name] += 1

    tf.test.main()
