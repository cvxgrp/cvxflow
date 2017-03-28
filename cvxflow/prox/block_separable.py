
from tensorflow.python.framework import ops

from cvxflow.prox import prox_function
from cvxflow.prox import block_ops


class BlockSeparable(prox_function.ProxFunction):
  def __init__(self, prox=None, shape=None, name="BlockSeparable"):
    with ops.name_scope(name):
      self.slices = block_ops.get_slices(shape)
      self.prox = prox
      super(BlockSeparable, self).__init__(name=name)

  def _call(self, v):
    return block_ops.vec(
      [prox_i(v_i) for prox_i, v_i
       in zip(self.prox, block_ops.to_list(v, self.slices))])
