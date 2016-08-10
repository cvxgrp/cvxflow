"""Represents a cone problem over tensors.

In particular, problems of the form
  minimize    c'x
  subject to  b - Ax in K
"""

from collections import namedtuple
from cvxpy.lin_ops import lin_utils
from cvxpy.lin_ops import tree_mat
import cvxpy as cvx
import tensorflow as tf

from cvxflow import cones
from cvxflow import cvxpy_expr
from cvxflow.cvxpy_expr import sum_dicts
from cvxflow.tf_util import vec, mat, vstack

CONE_MAP = {
    cvx.lin_ops.lin_constraints.LinEqConstr: cones.ZERO,
    cvx.lin_ops.lin_constraints.LinLeqConstr: cones.NONNEGATIVE,
    "": cones.SECOND_ORDER,
    "": cones.EXPONENTIAL,
    "": cones.SEMIDEFINITE,
}

VariableInfo = namedtuple("VariableInfo", ["id", "slice", "size"])
ConstraintInfo = namedtuple(
    "ConstraintInfo", ["id", "slice", "size", "cone"])

def get_var_info(obj, constrs):
    vars_ = lin_utils.get_expr_vars(obj)
    for constr in constrs:
        vars_ += lin_utils.get_expr_vars(constr.expr)
    vars_ = list(set(vars_))

    info = []
    offset = 0
    for var_id, var_size in vars_:
        size = var_size[0]*var_size[1]
        info.append(VariableInfo(
            var_id, slice(offset, offset+size), var_size))
        offset += size

    return info, offset

def get_constr_info(constrs):
    info = []
    offset = 0
    for constr in constrs:
        size = constr.size[0]*constr.size[1]
        info.append(ConstraintInfo(
            constr.constr_id, slice(offset, offset+size), constr.size,
            CONE_MAP[type(constr)]))
        offset += size

    return info, offset

class TensorProblem(object):
    def __init__(self, cvx_problem):
        obj, constrs = cvx_problem.canonicalize()

        self.var_info, self.n = get_var_info(obj, constrs)
        self.constr_info, self.m = get_constr_info(constrs)

        # TODO(mwytock): Need to fix signs here, likely we need to specify the
        # sign by cone type, e.g. Ax + b <= 0 will need to translated to
        # (-b) - Ax >= 0 to put it in the cone form we expect.
        self.A_exprs = [
            constr.expr for constr in tree_mat.prune_constants(constrs)]
        self.b = vstack(
            [vec(tf.constant(-tree_mat.mul(constr.expr, {})))
             for constr in constrs])

        # get c elements via gradient of c'x
        xs = [tf.Variable(tf.zeros(var.size, dtype=tf.float64))
              for var in self.var_info]
        obj_t = cvxpy_expr.tensor(
            obj, dict(zip((var.id for var in self.var_info), xs)))
        self.c = vstack([vec(ci) for ci in tf.gradients(obj_t, xs)])

    def A(self, x):
        xs = {var.id: mat(x[var.slice,:], var.size) for var in self.var_info}
        return vstack([vec(cvxpy_expr.tensor(Ai, xs)) for Ai in self.A_exprs])

    def AT(self, y):
        ys = [mat(y[c.slice,:], c.size) for c in self.constr_info]
        x_map = sum_dicts(cvxpy_expr.adjoint_tensor(Ai, ys[i])
                          for i, Ai in enumerate(self.A_exprs))
        return vstack([vec(x_map[var.id]) for var in self.var_info])
