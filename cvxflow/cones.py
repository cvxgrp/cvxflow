
import tensorflow as tf

from cvxflow.tf_util import vstack

ZERO = "zero"
NONNEGATIVE = "nonnegative"
SECOND_ORDER = "second_order"
EXPONENTIAL = "exponential"
SEMIDEFINITE = "semidefinite"

def proj_nonnegative(x):
    return tf.maximum(x, tf.zeros_like(x))
proj_dual_nonnegative = proj_nonnegative

def proj_second_order(x):
    pass
proj_dual_second_order = proj_second_order

def proj_semidefinite(x):
    pass
proj_dual_semidefinite = proj_semidefinite

def proj_zero(x):
    return tf.zeros_like(x)

def proj_dual_zero(x):
    return x

def proj_exponential(x):
    pass

def proj_dual_exponential(x):
    pass

def proj_cone(cone_slices, x, dual=False):
    ys = []
    prefix = "proj_dual_" if dual else "proj_"
    for cone, idx in cone_slices:
        proj = globals()[prefix + cone]
        ys.append(proj(x[idx,:]))

    return vstack(ys)
