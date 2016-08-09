
from collections import namedtuple
from cvxpy.lin_ops import tree_mat
import cvxpy as cvx
import operator
import tensorflow as tf

from cvxflow import cvxpy_expr

TF_ZERO = tf.constant(0)

def proj_nonnegative(x):
    return tf.maximum(x, tf.zeros_like(x))

def create_var(size, name):
    return tf.Variable(tf.zeros(size, dtype=tf.float64), name=name)

class TensorDict(object):
    def __init__(self, values):
        self.values = values

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if other == 0:
            return self
        return self.apply(operator.add, other)

    def __sub__(self, other):
        return self.apply(operator.sub, other)

    def apply(self, op, other):
        return TensorDict(
            {k: op(self.values.get(k, TF_ZERO), other.values.get(k, TF_ZERO))
             for k in self.values.viewkeys() | other.values.viewkeys()})

class TensorProblem(object):
    def __init__(self, cvx_problem):
        obj, constrs = cvx_problem.canonicalize()

        self.x = TensorDict({var.id: create_var(var.size, "x" + str(var.id))
                             for var in cvx_problem.variables()})
        self.obj = cvxpy_expr.tensor(obj, self.x.values)

        self.A_exprs = {
            constr.constr_id: constr.expr
            for constr in tree_mat.prune_constants(constrs)}
        self.b = TensorDict({
            constr.constr_id: tf.constant(tree_mat.mul(constr.expr, {}))
            for constr in constrs})
        self.c = TensorDict(dict(
            zip(self.x.values.keys(),
                tf.gradients(self.obj, self.x.values.values()))))

    def A(self, x):
        return TensorDict({constr_id: cvxpy_expr.tensor(Ai, x.values)
                           for constr_id, Ai in self.A_exprs.items()})

    def AT(self, y):
        return sum(TensorDict(cvxpy_expr.adjoint_tensor(Ai, y.values[constr_id]))
                   for constr_id, Ai in self.A_exprs.items())
