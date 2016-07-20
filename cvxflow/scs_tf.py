"""Solve an LP using SCS on tensorflow."""

from collections import namedtuple
import cvxpy as cvx
import numpy as np
import tensorflow as tf

PrimalVars = namedtuple("PrimalVars", ["x", "y", "tau"])
DualVars = namedtuple("DualVars", ["r", "s", "kappa"])

def dot(x, y):
    return tf.matmul(x, y, transpose_a=True)

def norm(x):
    return tf.sqrt(dot(x, x))

def proj_nonnegative(x):
    return tf.maximum(x, tf.zeros_like(x))

def proj_dual_cone(x, dims):
    idx = 0
    values = []

    # zero cone
    values += [x[idx:idx+dims["f"],0:1]]
    idx += dims["f"]

    # nonnegative orthant
    values += [proj_nonnegative(x[idx:idx+dims["l"],0:1])]
    idx += dims["l"]

    # TODO(mwytock): implement other cone projections

    return tf.concat(0, values)

def solve_linear(A, L, w_x, w_y):
    rhs = w_x - tf.matmul(A, w_y, transpose_a=True)
    z_x = tf.cholesky_solve(L, rhs)
    z_y = w_y + tf.matmul(A, z_x)
    return z_x, z_y

def subspace_projection(A, b, c, u, u_tilde, v):
    """u_tilde: primal update, linear solve."""

    # cache factorization of M and M^{-1}h
    n = A.get_shape()[1]
    ATA = tf.matmul(A, A, transpose_a=True)
    I = tf.constant(np.eye(n), dtype=tf.float32)
    L = tf.cholesky(I + ATA)
    g_x, g_y = solve_linear(A, L, c, b)
    g_dot_h = dot(g_x, c) + dot(g_y, b)

    # u_tilde: solve linear system
    w_x = u.x + v.r
    w_y = u.y + v.s
    w_tau = u.tau + v.kappa

    z_x, z_y = solve_linear(A, L, w_x, w_y)
    g_dot_w = dot(g_x, w_x) + dot(g_y, w_y)
    alpha = (w_tau*g_dot_h - dot(z_x, c) - dot(z_y, b))/(1 + g_dot_h) - w_tau

    u_tilde_x = z_x + alpha*g_x
    u_tilde_y = z_y + alpha*g_y

    # u_tilde, primal update, linear solve
    return tf.group(
        u_tilde.x.assign(u_tilde_x),
        u_tilde.y.assign(u_tilde_y),
        u_tilde.tau.assign(w_tau + dot(c, u_tilde_x) + dot(b, u_tilde_y)))

def cone_projection(dims, u, u_tilde, v):
    return tf.group(
        u.x.assign(u_tilde.x - v.r),
        u.y.assign(proj_dual_cone(u_tilde.y - v.s, dims)),
        u.tau.assign(proj_nonnegative(u_tilde.tau - v.kappa)))

def dual_update(u, u_tilde, v):
    return tf.group(
        v.r.assign(v.r - u_tilde.x + u.x),
        v.s.assign(v.s - u_tilde.y + u.y),
        v.kappa.assign(v.kappa - u_tilde.tau + u.tau))

def residuals(A, b, c, u, v):
    """residuals and duality gap."""
    x = u.x / u.tau
    y = u.y / u.tau
    s = v.s / u.tau
    p_norm = norm(tf.matmul(A, x) + s - b)
    d_norm = norm(tf.matmul(A, y, transpose_a=True) + c)
    c_dot_x = dot(c, x)
    b_dot_y = dot(b, y)
    return p_norm, d_norm, c_dot_x, b_dot_y, x, y, s

if __name__ == "__main__":
    # Form LP and get SCS form with cvxpy
    _m = 5
    _n = 10

    np.random.seed(0)
    A = np.abs(np.random.randn(_m,_n))
    b = A.dot(np.abs(np.random.randn(_n)))
    c = np.random.rand(_n) + 0.5

    x = cvx.Variable(_n)
    prob = cvx.Problem(cvx.Minimize(c.T*x), [A*x == b, x >= 0])
    data = prob.get_problem_data(cvx.SCS)

    # create tensorflow graph and solve

    # inputs
    m, n = data["A"].shape
    A = tf.placeholder(tf.float32, shape=(m, n))
    b = tf.placeholder(tf.float32, shape=(m, 1))
    c = tf.placeholder(tf.float32, shape=(n, 1))

    # variables
    u = PrimalVars(
        tf.Variable(tf.expand_dims(tf.zeros(n), 1)),
        tf.Variable(tf.expand_dims(tf.zeros(m), 1)),
        tf.Variable(tf.expand_dims(tf.ones(1), 1)))
    u_tilde = PrimalVars(
        tf.Variable(tf.expand_dims(tf.zeros(n), 1)),
        tf.Variable(tf.expand_dims(tf.zeros(m), 1)),
        tf.Variable(tf.expand_dims(tf.ones(1), 1)))
    v = DualVars(
        tf.Variable(tf.expand_dims(tf.zeros(n), 1)),
        tf.Variable(tf.expand_dims(tf.zeros(m), 1)),
        tf.Variable(tf.expand_dims(tf.ones(1), 1)))

    # ops
    init = tf.initialize_all_variables()
    subspace_projection_op = subspace_projection(A, b, c, u, u_tilde, v)
    cone_projection_op = cone_projection(data["dims"], u, u_tilde, v)
    dual_update_op = dual_update(u, u_tilde, v)
    p_norm, d_norm, c_dot_x, b_dot_y, x, y, s = residuals(A, b, c, u, v)

    # solve with tensorflow
    feed_dict = {
        A: data["A"].todense(),
        b: data["b"].reshape(-1,1),
        c: data["c"].reshape(-1,1),
    }
    b_norm = np.linalg.norm(data["b"])
    c_norm = np.linalg.norm(data["c"])
    max_iterations = 200

    with tf.Session() as sess:
        sess.run(init)
        for k in xrange(max_iterations):
            # run primal/dual updates
            sess.run(subspace_projection_op, feed_dict=feed_dict)
            sess.run(cone_projection_op)
            sess.run(dual_update_op)

            # compute residuals
            if k % 20 == 0:
                p_norm0, d_norm0, c_dot_x0, b_dot_y0, tau0, kappa0 = sess.run(
                    [p_norm, d_norm, c_dot_x, b_dot_y, u.tau, v.kappa],
                    feed_dict=feed_dict)

                g = c_dot_x0 + b_dot_y0
                print "k=%d, ||p||=%.4e, ||d||=%.4e, |g|=%.4e, tau=%.4e, kappa=%.4e" % (
                    k,
                    p_norm0 / (1 + b_norm),
                    d_norm0 / (1 + c_norm),
                    np.abs(g) / (1 + np.abs(c_dot_x0) + np.abs(b_dot_y0)),
                    tau0, kappa0)
    print c_dot_x0

    # solve with SCS
    prob.solve(solver=cvx.SCS, verbose=True)
