
from collections import namedtuple
from cvxpy.lin_ops import tree_mat
import cvxpy as cvx
import operator
import tensorflow as tf

from cvxflow import cvxpy_expr
from cvxflow.tf_util import vec, mat, vstack

def get_slices(elems, key):
    offset = 0
    slices = {}
    for x in elems:
        size = x.size[0]*x.size[1]
        slices[key(x)] = slice(offset, offset+size)
        offset += size
    return slices

def get_vars(obj, constrs):
    vars_ = lin_utils.get_expr_vars(obj)
    for constr in constrs:
        vars_ += lin_utils.get_expr_vars(constr)
    return list(set(vars_))

class TensorProblem(object):
    def __init__(self, cvx_problem):
        obj, constrs = cvx_problem.canonicalize()

        self.vars = get_vars(obj, constrs)
        self.var_slices = get_slices(self.vars, lambda x: x.data)
        self.constrs = constrs
        self.constr_slices = get_slices(self.constrs, lambda x: x.constr_id)

        self.A_exprs = {
            constr.constr_id: constr.expr
            for constr in tree_mat.prune_constants(constrs)}
        self.b = vstack(
            [vec(tf.constant(tree_mat.mul(constr.expr, {})))
             for constr in constrs])

        # get c elements by taking gradient of c'x
        x = [tf.Variable(tf.zeros(var.size, dtype=tf.float64))]
        obj = cvxpy_expr.tensor(
            obj, dict(zip((var.id for var in cvx_problem.variables()))))
        self.c = vstack([vec(ci) for ci in tf.gradients(obj, x)])

    def A(self, x):
        xs = {var.id: tf.reshape(x[self.var_slices[var_id]], var.size)
              for var in self.vars}
        return vstack([vec(cvxpy_expr.tensor(Ai, xs)) for Ai in self.A_exprs])

    def AT(self, y):
        ys = [mat(y[self.constr_slices[constr.constr_id]], constr.size)
              for constr in self.constrs]
        x_map = cvxpy_expr.sum_dicts(
            cvxpy_expr.adjoint_tensor(Ai, ys[i])
            for i, Ai in enumerate(self.A_exprs))
        return vstack(vec(x_map[var.id]) for var in self.vars)
