
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops

from cvxflow.prox import admm
from cvxflow import conjugate_gradient

def to_list(x, slices):
  return [array_ops.reshape(x[idx], shape) for idx, shape in slices]

def vec(x_list):
  return array_ops.concat(
    [array_ops.reshape(x, (-1, 1)) for x in x_list], axis=0)

def _get_slices(shapes):
  offset = 0
  slices = []
  for shape in shapes:
    size = np.prod(shape)
    slices.append((slice(offset, size), shape))
    offset += size
  return slices, offset

class POGS(object):
  """Proximal Operator Graph Solver.

  minimize    f(y) + g(x)
  subject to  y = Ax
  """
  def __init__(
      self, prox_f=None, prox_g=None, A=None, shape=None, dtype=dtypes.float32,
      name="POGS"):
    with ops.name_scope(name):
      self.prox_f = prox_f or [lambda x: x]
      self.prox_g = prox_g or [lambda x: x]
      self.y_slice, y_size = _get_slices(shape[0])
      self.x_slice, x_size = _get_slices(shape[1])
      self.A, self.AT = A
      self.y_size = y_size
      n = x_size + y_size
      self.admm = admm.ADMM(
        self._apply_prox, self._project, dtype=dtype, shape=(n, n, n, 1))

  def _split_x_y(self, xy):
    return xy[:self.y_size], xy[self.y_size:]

  def _x_list(self, x):
    return to_list(x, self.x_slice)

  def _y_list(self, y):
    return to_list(y, self.y_slice)

  def _apply_prox(self, v):
    v_y, v_x = self._split_x_y(v)
    return vec([prox(v) for prox, v in zip(self.prox_f, self._y_list(v_y))] +
               [prox(v) for prox, v in zip(self.prox_g, self._x_list(v_x))])

  def _project(self, v):
    v_y, v_x = self._split_x_y(v)

    # TODO(mwytock): use CGLS rather than CG here
    def I_ATA(x):
      return x + vec(self.AT(self.A(self._x_list(x))))
    x = conjugate_gradient.conjugate_gradient_solve(
      I_ATA,
      v_x + vec(self.AT(self._y_list(v_y))),
      v_x)[0]

    y = self.A(self._x_list(x))
    return vec(y + [x])

  def solve(self, **kwargs):
    y, x = self._split_x_y(self.admm.x)
    y_list = self._y_list(y)
    x_list = self._x_list(x)
    kwargs["output"] = y_list + x_list
    variables = self.admm.solve(**kwargs)
    return variables[:len(y_list)], variables[len(y_list):]
