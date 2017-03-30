
import tensorflow as tf

def dot(a, b):
    with tf.name_scope("dot"):
        return tf.reduce_sum(a*b)

def vec(X):
    with tf.name_scope("vec"):
        return tf.reshape(X, [-1, 1])

def mat(x, size):
    with tf.name_scope("mat"):
        return tf.reshape(x, size)

def vstack(xs):
    with tf.name_scope("vstack"):
        return tf.concat(xs, axis=0)
