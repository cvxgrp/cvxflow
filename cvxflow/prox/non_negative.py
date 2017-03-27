

from tensorflow.python.ops import math_ops

from cvxflow.prox import prox_function

class NonNegative(prox_function.ProxFunction):
  def __init__(self, name="NonNegative"):
    super(NonNegative, self).__init__(name=name)

  def _call(self, v):
    return math_ops.maximum(v, 0)
