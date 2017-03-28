
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def get_slices(shape):
  offset = 0
  slices = []
  for shape_i in shape:
    size_i = np.prod(shape_i)
    slices.append((slice(offset, offset+size_i), shape_i))
    offset += size_i
  return slices


def to_list(x, slices, name="to_list"):
  with ops.name_scope(name, values=[x]):
    return [array_ops.reshape(x[idx], shape) for idx, shape in slices]

def to_vector(x_list, name="vec"):
  with ops.name_scope(name, values=x_list):
    return array_ops.concat(
      [array_ops.reshape(x, (-1, 1)) for x in x_list], axis=0)
