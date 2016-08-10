
from collections import namedtuple
from cvxpy.lin_ops import lin_utils
from cvxpy.lin_ops import tree_mat
import cvxpy as cvx
import tensorflow as tf

from cvxflow import cvxpy_expr
from cvxflow.tf_util import vec, mat, vstack

def get_slices(x_sizes):
    offset = 0
    slices = {}
    for x_id, x_size in x_sizes:
        size = x_size[0]*x_size[1]
        slices[x_id] = slice(offset, offset+size)
        offset += size
    return slices

def get_var_sizes(obj, constrs):
    vars_ = lin_utils.get_expr_vars(obj)
    for constr in constrs:
        vars_ += lin_utils.get_expr_vars(constr.expr)
    return list(set(vars_))

def get_constr_sizes(constrs):
    return [(constr.constr_id, constr.size) for constr in constrs]

class TensorProblem(object):
    def __init__(self, cvx_problem):
        obj, constrs = cvx_problem.canonicalize()

        self.var_sizes = get_var_sizes(obj, constrs)
        self.var_slices = get_slices(self.var_sizes)
        self.constr_sizes = get_constr_sizes(constrs)
        self.constr_slices = get_slices(self.constr_sizes)

        # TODO(mwytock): Verify signs on A, b here
        self.A_exprs = [
            constr.expr for constr in tree_mat.prune_constants(constrs)]
        self.b = vstack(
            [vec(tf.constant(-tree_mat.mul(constr.expr, {})))
             for constr in constrs])

        # get c elements via gradient of c'x
        xs = [tf.Variable(tf.zeros(var_size, dtype=tf.float64))
              for _, var_size in self.var_sizes]
        obj_t = cvxpy_expr.tensor(
            obj, dict(zip((var_id for var_id, _ in self.var_sizes), xs)))
        self.c = vstack([vec(ci) for ci in tf.gradients(obj_t, xs)])

    def A(self, x):
        xs = {var_id: mat(x[self.var_slices[var_id],0], var_size)
              for var_id, var_size in self.var_sizes}
        return vstack([vec(cvxpy_expr.tensor(Ai, xs)) for Ai in self.A_exprs])

    def AT(self, y):
        ys = [mat(y[self.constr_slices[constr_id],0], constr_size)
              for constr_id, constr_size in self.constr_sizes]
        x_map = cvxpy_expr.sum_dicts(
            cvxpy_expr.adjoint_tensor(Ai, ys[i])
            for i, Ai in enumerate(self.A_exprs))
        return vstack([vec(x_map[var_id]) for var_id, _ in self.var_sizes])
