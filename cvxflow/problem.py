"""Represents a cone problem over tensors.

In particular, problems of the form
  minimize    c'x
  subject to  b - Ax in K
"""

from collections import namedtuple
from cvxpy.lin_ops import lin_op
from cvxpy.lin_ops import lin_utils
from cvxpy.lin_ops import tree_mat
from cvxpy.problems.solvers.utilities import SOLVERS
from cvxpy.problems.problem_data.sym_data import SymData
import cvxpy.settings as s
import cvxpy as cvx
import numpy as np
import tensorflow as tf

from cvxflow import cones
from cvxflow import cvxpy_expr
from cvxflow.cvxpy_expr import sum_dicts
from cvxflow.tf_util import vec, mat, vstack


def get_constraint_tensors(constraints):
    """Get expression for Ax + b."""
    A_exprs = [constr.expr for constr in tree_mat.prune_constants(constraints)]
    b = vstack(
        [vec(tf.constant(-tree_mat.mul(constr.expr, {}), dtype=tf.float32))
         for constr in constraints])
    return A_exprs, b

def get_objective_tensor(var_ids, sym_data):
    """Get objective tensor via gradient of c'x."""
    xs = [tf.Variable(tf.zeros(sym_data.var_sizes[var_id], dtype=tf.float32))
          for var_id in var_ids]
    xs_map = dict(zip((var_id for var_id in var_ids), xs))
    obj_t = cvxpy_expr.tensor(sym_data.objective, xs_map)

    # get gradient, handling None values
    return vstack([
        vec(ci) if ci is not None
        else vec(tf.zeros(sym_data.var_sizes[var_ids[i]], dtype=tf.float32))
        for i, ci in enumerate(tf.gradients(obj_t, xs))])

class TensorProblem(object):
    def __init__(self, cvx_problem):
        objective, constraints = cvx_problem.canonicalize()
        self.sym_data = SymData(objective, constraints, SOLVERS[s.SCS])

        # sort variables by offset
        self.var_ids = [var_id for var_id, var_offset in sorted(
            self.sym_data.var_offsets.items(),
            key=lambda (var_id, var_offset): var_offset)]
        self.constraints = (self.sym_data.constr_map[s.EQ] +
                            self.sym_data.constr_map[s.LEQ])

        self.A_exprs, self.b = get_constraint_tensors(self.constraints)
        self.c = get_objective_tensor(self.var_ids, self.sym_data)

    def A(self, x):
        xs = {}
        for var_id, var_size in self.sym_data.var_sizes.items():
            var_offset = self.sym_data.var_offsets[var_id]
            idx = slice(var_offset, var_offset+var_size[0]*var_size[1])
            xs[var_id] = mat(x[idx,:], var_size)
        return vstack([vec(cvxpy_expr.tensor(Ai, xs)) for Ai in self.A_exprs])

    def AT(self, y):
        ys = []
        offset = 0
        for constr in self.constraints:
            idx = slice(offset, offset+constr.size[0]*constr.size[1])
            ys.append(mat(y[idx,:], constr.size))
            offset += constr.size[0]*constr.size[1]
        x_map = sum_dicts(cvxpy_expr.adjoint_tensor(Ai, ys[i])
                          for i, Ai in enumerate(self.A_exprs))
        return vstack([vec(x_map[var_id]) for var_id in self.var_ids])

    @property
    def cone_slices(self):
        dims = self.sym_data.dims
        retval = []
        i = 0

        if dims[s.EQ_DIM]:
            retval.append((cones.ZERO, slice(i, i+dims[s.EQ_DIM])))
            i += dims[s.EQ_DIM]

        if dims[s.LEQ_DIM]:
            retval.append((cones.NONNEGATIVE, slice(i, i+dims[s.LEQ_DIM])))
            i += dims[s.LEQ_DIM]

        for qi in dims[s.SOC_DIM]:
            retval.append((cones.SECOND_ORDER, slice(i, i+qi)))
            i += qi

        return retval
