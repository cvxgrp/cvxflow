

from tensorflow.python.platform import test

from cvxflow.prox import non_negative
from cvxflow.prox import prox_function_testutil


class NonNegativeTest(prox_function_testutil.ProxFunctionTest):
  def _get_prox_function_for_dtype(self, dtype):
    return non_negative.NonNegative()

  def testBasic(self):
    self._verify([1,-2,0], [1,0,0])


if __name__ == "__main__":
  test.main()
