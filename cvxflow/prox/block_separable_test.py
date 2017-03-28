
from tensorflow.python.platform import test
from tensorflow.python.ops import math_ops

from cvxflow.prox import block_separable
from cvxflow.prox import prox_function_testutil

class BlockSeparableTest(prox_function_testutil.ProxFunctionTest):
  def _get_prox_function_for_dtype(self, dtype, prox=None, shape=None):
    return block_separable.BlockSeparable(prox=prox, shape=shape)

  def testBasic(self):
    self._verify([[-2], [2], [-2], [2]], [[-2], [2], [0], [2]],
                 prox=[lambda x: x,
                       lambda x: math_ops.maximum(x, 0)],
                 shape=[(1,2),(2,1)])

if __name__ == "__main__":
  test.main()
