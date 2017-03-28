
import tensorflow as tf

def dot(a, b):
    return tf.squeeze(tf.matmul(a, b, transpose_a=True))

def vec(X):
    return tf.reshape(X, [-1, 1])

def mat(x, size):
    return tf.reshape(x, size)

def vstack(xs):
    return tf.concat(xs, axis=0)
