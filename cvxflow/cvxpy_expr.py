"""Convert cvxpy LinOp to tensorflow graphs."""

import numpy as np
import cvxpy as cvx
import tensorflow as tf


def to_graph_mul(lin_op, var_map):
    return tf.matmul(
        to_graph(lin_op.data, var_map),
        to_graph(lin_op.args[0], var_map))

def to_graph_sum(lin_op, var_map):
    return tf.add(
        to_graph(lin_op.args[0], var_map),
        to_graph(lin_op.args[1], var_map))

def to_graph_neg(lin_op, var_map):
    return tf.neg(
        to_graph(lin_op.args[0], var_map))

def to_graph_promote(lin_op, var_map):
    # tensorflow handles promotion automatically
    return to_graph(lin_op.args[0], var_map)

def to_graph_dense_const(lin_op, var_map):
    return tf.constant(lin_op.data, dtype=tf.float64)

def to_graph_scalar_const(lin_op, var_map):
    return tf.constant(lin_op.data, dtype=tf.float64)

def to_graph_variable(lin_op, var_map):
    var_id = lin_op.data
    if var_id not in var_map:
        var_map[var_id] = tf.Variable(
            tf.zeros(lin_op.size, dtype=tf.float64), name="var" + str(var_id))
    return var_map[var_id]

def to_graph(lin_op, var_map):
    f_name = "to_graph_" + lin_op.type
    return globals()[f_name](lin_op, var_map)

if __name__ == "__main__":
    m = 5
    n = 10
    A = np.random.randn(m,n)
    b = A.dot(np.abs(np.random.randn(n)))
    c = np.random.rand(n)
    x = cvx.Variable(n)
    prob = cvx.Problem(cvx.Minimize(c.T*x), [A*x == b, x >= 0])

    var_map = {}
    obj, constrs =  prob.canonicalize()
    obj_tf = tf.squeeze(to_graph(obj, var_map), name="obj")
    constrs_tf = [
        tf.squeeze(to_graph(constr.expr, var_map), name="constr" + str(i))
        for i, constr in enumerate(constrs)]

    init = tf.initialize_all_variables()
    tf.scalar_summary("obj_val", obj_tf)
    merged = tf.merge_all_summaries()
    with tf.Session() as sess:
        sess.run(init)
        summary = sess.run(merged)
        writer = tf.train.SummaryWriter(".", sess.graph)
        writer.add_summary(summary, 0)
