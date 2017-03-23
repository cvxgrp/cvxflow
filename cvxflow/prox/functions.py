
import contextlib

from tensorflow.contrib import framework as contrib_framework
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

class ProxFunction(object):
    def __init__(self,
                 graph_parents=None,
                 name=None):
        graph_parents = [] if graph_parents is None else graph_parents
        for i, t in enumerate(graph_parents):
            if t is None or not contrib_framework.is_tensor(t):
                raise ValueError("Graph parent item %d is not a Tensor; %s." % (i, t))
        self._graph_parents = graph_parents
        self._name = name or type(self).__name__

    @contextlib.contextmanager
    def _name_scope(self, name=None, values=None):
        """Helper function to standardize op scope."""
        with ops.name_scope(self.name):
            with ops.name_scope(name, values=(
                    ([] if values is None else values) + self._graph_parents)) as scope:
                yield scope

    @property
    def name(self):
        return self._name

    @property
    def graph_parents(self):
        return self._graph_parents

    def _call(self, v):
        raise NotImplementedError

    def __call__(self, v, name="__call__"):
        with self._name_scope(name, values=[v]):
            return self._call(v)


class AbsoluteValue(ProxFunction):
    """Absolute value.

    If lam is a scalar, lam*|x|
    If lam is a tuple, -lam[0]*neg(x) + lam[1]*pos(x)."""
    def __init__(self,
                 scale,
                 name="AbsoluteValue"):
        with ops.name_scope(name, values=[scale]):
            if isinstance(scale, tuple):
                assert len(scale) == 2
                self.scale_neg = ops.convert_to_tensor(scale[0], name="scale_neg")
                self.scale_pos = ops.convert_to_tensor(scale[1], name="scale_pos")
            else:
                self.scale_neg = ops.convert_to_tensor(scale, name="scale")
                self.scale_pos = self.scale_neg

            super(AbsoluteValue, self).__init__(
                graph_parents=[self.scale_neg, self.scale_pos],
                name=name)

    def _call(self, v):
        return (math_ops.maximum(v - self.scale_pos, 0) +
                math_ops.minimum(v + self.scale_neg, 0))


class LeastSquares(ProxFunction):
    """Least squares.

    0.5*sum_squares(A*x - b) + 0.5*mu*sum_squares(x)."""
    def __init__(self,
                 A=None,
                 b=None,
                 mu=None,
                 W=None,
                 name="LeastSquares"):
        with ops.name_scope(name, values=[A, b, mu, W]):
            graph_parents = []
            n = None
            if A is not None:
                graph_parents.extend(A.graph_parents)
                n = int(A.shape[1])
                dtype = A.dtype
                A = A.to_dense()
            elif W is not None:
                graph_parents.extend(W.graph_parents)
                n = int(W.shape[1])
                dtype = W.dtype
                W = W.to_dense()
            else:
                raise ValueError("Must specify either A or W")

            if b is not None:
                b = ops.convert_to_tensor(b)
                graph_parents.append(b)
            if mu is not None:
                mu = ops.convert_to_tensor(mu)
                graph_parents.append(mu)


            M = 0
            if A is not None:
                M += math_ops.matmul(A, A, transpose_a=True)
            if A is not None and b is not None:
                self.ATb = math_ops.matmul(A, b, transpose_a=True)
            else:
                self.ATb = 0

            if W is not None:
                M += math_ops.matmul(W, W, transpose_a=True)
            else:
                M += linalg_ops.eye(n, dtype=dtype)

            if mu is not None:
                M += mu*linalg_ops.eye(n, dtype=dtype)

            self.L = linalg_ops.cholesky(M)

            super(LeastSquares, self).__init__(
                graph_parents=graph_parents,
                name=name)

    def _call(self, v):
        return linalg_ops.cholesky_solve(self.L, self.ATb + v)
