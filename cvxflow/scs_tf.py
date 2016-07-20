"""Solve an LP using SCS on tensorflow."""

from collections import namedtuple
import time

import cvxpy as cvx
import numpy as np
import tensorflow as tf

PrimalVars = namedtuple("PrimalVars", ["x", "y", "tau"])
DualVars = namedtuple("DualVars", ["r", "s", "kappa"])
Cache = namedtuple("Cache", ["L", "g_x", "g_y"])

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

def factorize(A, b, c, cache):
    """Factorize M and compute g = M^{-1}h."""
    n = A.get_shape()[1]
    ATA = tf.matmul(A, A, transpose_a=True)
    I = tf.constant(np.eye(n), dtype=tf.float32)
    L = tf.cholesky(I + ATA)
    g_x, g_y = solve_linear(A, L, c, b)

    return tf.group(
        cache.L.assign(L),
        cache.g_x.assign(g_x),
        cache.g_y.assign(g_y))

def iteration(A, b, c, dims, cache, u, v):
    """A single SCS iteration."""
    # u_tilde: solve linear system
    w_x = u.x + v.r
    w_y = u.y + v.s
    w_tau = u.tau + v.kappa

    z_x, z_y = solve_linear(A, cache.L, w_x, w_y)
    g_dot_w = dot(cache.g_x, w_x) + dot(cache.g_y, w_y)
    g_dot_h = dot(cache.g_x, c)   + dot(cache.g_y, b)
    alpha = (w_tau*g_dot_h - dot(z_x, c) - dot(z_y, b))/(1 + g_dot_h) - w_tau

    u_tilde_x = z_x + alpha*cache.g_x
    u_tilde_y = z_y + alpha*cache.g_y
    u_tilde_tau = w_tau + dot(c, u_tilde_x) + dot(b, u_tilde_y)

    # u: cone projection
    u_x = u_tilde_x - v.r
    u_y = proj_dual_cone(u_tilde_y - v.s, dims)
    u_tau = proj_nonnegative(u_tilde_tau - v.kappa)

    # v: dual update
    v_r = v.r - u_tilde_x + u_x
    v_s = v.s - u_tilde_y + u_y
    v_kappa = v.kappa - u_tilde_tau + u_tau

    return tf.group(
        u.x.assign(u_x),
        u.y.assign(u_y),
        u.tau.assign(u_tau),
        v.r.assign(v_r),
        v.s.assign(v_s),
        v.kappa.assign(v_kappa))

def residuals(A, b, c, u, v):
    """SCS residuals and duality gap."""
    x = u.x / u.tau
    y = u.y / u.tau
    s = v.s / u.tau
    p_norm = norm(tf.matmul(A, x) + s - b)
    d_norm = norm(tf.matmul(A, y, transpose_a=True) + c)
    c_dot_x = dot(c, x)
    b_dot_y = dot(b, y)
    return p_norm, d_norm, c_dot_x, b_dot_y, x, y, s

def solve_scs_tf(data, iterations):
    """Create SCS tensorflow graph and solve."""
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
    v = DualVars(
        tf.Variable(tf.expand_dims(tf.zeros(n), 1)),
        tf.Variable(tf.expand_dims(tf.zeros(m), 1)),
        tf.Variable(tf.expand_dims(tf.ones(1), 1)))

    # cache
    cache = Cache(
        tf.Variable(tf.zeros((n, n))),
        tf.Variable(tf.expand_dims(tf.zeros(n), 1)),
        tf.Variable(tf.expand_dims(tf.zeros(m), 1)))

    # ops
    init = tf.initialize_all_variables()
    factorize_op = factorize(A, b, c, cache)
    iteration_op = iteration(A, b, c, data["dims"], cache, u, v)
    p_norm, d_norm, c_dot_x, b_dot_y, x, y, s = residuals(A, b, c, u, v)

    # solve with tensorflow
    feed_dict = {
        A: data["A"].todense(),
        b: data["b"].reshape(-1,1),
        c: data["c"].reshape(-1,1),
    }
    b_norm = np.linalg.norm(data["b"])
    c_norm = np.linalg.norm(data["c"])

    if trace:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    t0 = time.time()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(factorize_op, feed_dict=feed_dict)

        for k in xrange(iterations):
            if trace:
                sess.run(iteration_op, feed_dict=feed_dict, options=run_options,
                         run_metadata=run_metadata)

                tl = tf.python.client.timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open("scs_tf_iter_%d.json" % k, "w") as f:
                    f.write(ctf)
            else:
                sess.run(iteration_op, feed_dict=feed_dict)

            if k % 20 == 0:
                # compute residuals
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
        print "objective value = %.4f" % c_dot_x0
        print "%.2e seconds" % (time.time() - t0)

if __name__ == "__main__":
    # form LP
    np.random.seed(0)
    m = 500
    n = 1000
    A = np.abs(np.random.randn(m,n))
    b = A.dot(np.abs(np.random.randn(n)))
    c = np.random.rand(n)
    x = cvx.Variable(n)
    prob = cvx.Problem(cvx.Minimize(c.T*x), [A*x == b, x >= 0])

    # solve with tensorflow
    iters = 10
    trace = True
    solve_scs_tf(prob.get_problem_data(cvx.SCS), iters)

    # solve with SCS
    t0 = time.time()
    prob.solve(solver=cvx.SCS, verbose=True)
    print "%.2e seconds" % (time.time() - t0)
