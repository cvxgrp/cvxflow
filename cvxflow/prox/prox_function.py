
import contextlib

from tensorflow.contrib import framework
from tensorflow.python.framework import ops

class ProxFunction(object):
  def __init__(self,
               graph_parents=None,
               name=None):
    self.graph_parents = [] if graph_parents is None else graph_parents
    self.name = name or type(self).__name__

  @contextlib.contextmanager
  def _name_scope(self, name=None, values=None):
    """Helper function to standardize op scope."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=(
          ([] if values is None else values) + self.graph_parents)) as scope:
        yield scope

  def _call(self, v):
    raise NotImplementedError

  def __call__(self, v, name="__call__"):
    with self._name_scope(name, values=[v]):
      return self._call(v)
