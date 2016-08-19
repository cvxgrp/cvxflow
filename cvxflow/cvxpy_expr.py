"""Convert cvxpy LinOp expression trees to TensorFlow graphs."""

import numpy as np
import cvxpy as cvx
import tensorflow as tf

def sum_dicts(dicts):
    sum_dict = {}
    for val_dict in dicts:
        for id_, value in val_dict.items():
            if id_ in sum_dict:
                sum_dict[id_] = sum_dict[id_] + value
            else:
                sum_dict[id_] = value
    return sum_dict

def is_scalar(x):
    for dim in x.get_shape():
        if dim != 1:
            return False
    return True

def is_sparse(x):
    return type(x) == tf.SparseTensor

def tensor(lin_op, value_map={}):
    f_name = "tensor_" + lin_op.type
    return globals()[f_name](lin_op, value_map)

def tensor_mul(lin_op, value_map):
    a = tensor(lin_op.data, value_map)
    b = tensor(lin_op.args[0], value_map)
    if is_sparse(a):
        return tf.sparse_tensor_dense_matmul(a, b)
    elif is_scalar(a) or is_scalar(b):
        return tf.mul(a, b)
    else:
        return tf.matmul(a, b)

def tensor_sum(lin_op, value_map):
    if len(lin_op.args) == 1:
        # special case for single arg sum
        return tensor(lin_op.args[0], value_map)

    return tf.add(
        tensor(lin_op.args[0], value_map),
        tensor(lin_op.args[1], value_map))

def tensor_neg(lin_op, value_map):
    return tf.neg(
        tensor(lin_op.args[0], value_map))

def tensor_promote(lin_op, value_map):
    # NOTE(mwytock): promotion handled directly in mul and add
    return tensor(lin_op.args[0], value_map)

def tensor_scalar_const(lin_op, value_map):
    return tf.constant(lin_op.data, dtype=tf.float32)

def tensor_dense_const(lin_op, value_map):
    return tf.constant(lin_op.data, dtype=tf.float32)

def tensor_sparse_const(lin_op, value_map):
    A = lin_op.data.tocoo()
    indices = tf.constant(np.vstack((A.row, A.col)).T, dtype=tf.int64)
    values = tf.constant(A.data, dtype=tf.float32)
    shape = tf.constant(A.shape, dtype=tf.int64)
    return tf.SparseTensor(indices, values, shape)

def tensor_variable(lin_op, value_map):
    var_id = lin_op.data
    return value_map[var_id]

def tensor_conv(lin_op, value_map):
    c = tensor(lin_op.data)
    x = tensor(lin_op.args[0], value_map)
    m = lin_op.data.size[0]
    n = lin_op.args[0].size[0]

    # add padding and flip
    c = tf.concat(0, [tf.zeros((n-1, 1), dtype=tf.float32), c])
    x = tf.concat(0, [x, tf.zeros((m-1, 1), dtype=tf.float32)])
    c = tf.reverse(c, dims=[True, False])

    return tf.reshape(
        tf.nn.conv2d(
            tf.reshape(x, (1, 1, m+n-1, 1)),
            tf.reshape(c, (1, m+n-1, 1, 1)),
            strides=[1, 1, 1, 1],
            padding="SAME"),
        (m+n-1, 1))

def tensor_sum_entries(lin_op, value_map):
    return tf.reduce_sum(tensor(lin_op.args[0], value_map), keep_dims=True)

def adjoint_tensor(lin_op, value):
    f_name = "adjoint_tensor_" + lin_op.type
    return globals()[f_name](lin_op, value)

def adjoint_tensor_mul(lin_op, value):
    a = tensor(lin_op.data)
    b = value

    if is_sparse(a):
        c = tf.sparse_tensor_dense_matmul(a, b, adjoint_a=True)
    elif is_scalar(a) or is_scalar(b):
        c = tf.mul(tf.transpose(a), b)
    else:
        c = tf.matmul(a, b, transpose_a=True)

    return adjoint_tensor(lin_op.args[0], c)

def adjoint_tensor_neg(lin_op, value):
    return adjoint_tensor(lin_op.args[0], -value)

def adjoint_tensor_sum(lin_op, value):
    return sum_dicts(adjoint_tensor(arg, value) for arg in lin_op.args)

def adjoint_tensor_variable(lin_op, value):
    var_id = lin_op.data
    return {var_id: value}

def adjoint_tensor_conv(lin_op, value):
    c = tensor(lin_op.data)
    m = lin_op.data.size[0]
    n = lin_op.args[0].size[0]

    return adjoint_tensor(
        lin_op.args[0],
        tf.reshape(
            tf.nn.conv2d(
                tf.reshape(value, (1, 1, m+n-1, 1)),
                tf.reshape(c, (1, m, 1, 1)),
                strides=[1, 1, 1, 1],
                padding="VALID"),
            (n, 1)))
