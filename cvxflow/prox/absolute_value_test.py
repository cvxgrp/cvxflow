

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

from cvxflow.prox import prox_function_testutil
from cvxflow.prox import absolute_value

class AbsoluteValueTest(prox_function_testutil.ProxFunctionTest):
    def _get_prox_function_for_dtype(self, dtype, scale=None):
        if isinstance(scale, tuple):
            scale = (ops.convert_to_tensor(scale[0], dtype=dtype),
                     ops.convert_to_tensor(scale[1], dtype=dtype))
        elif scale is not None:
            scale = ops.convert_to_tensor(scale, dtype=dtype)

        return absolute_value.AbsoluteValue(scale)

    def testDefault(self):
        self._verify([-3.,-1.,4.], [-2.,0.,3.], scale=1.)
    def testScaleSymmetric(self):
        self._verify([-3.,-1.,4.], [-1.,0.,2.], scale=2.)
    def testScaleAsymmetric(self):
        self._verify([-4.,-2.,-1.,2.,3.], [-1.,0.,0.,1.,2.], scale=(3.,1.))
    def testScale2d(self):
        self._verify([[-3,-1,4.], [-3,-1,4.]],
                     [[0,0,3.], [-2,0,1.]],
                     scale=([[3],[1]],[[1],[3]]))

if __name__ == "__main__":
    test.main()
