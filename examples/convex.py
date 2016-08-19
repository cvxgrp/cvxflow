"""Examples of solving convex problems with TensorFlow."""

import cvxpy as cvx
import numpy as np
import sys
import time

def nn_deconv(n):
    sigma = 0.1*n
    k = 5

    x0 = np.zeros(n)
    x0[np.random.choice(n,k)] = np.random.randn(k)*n/10

    c = np.exp(-np.arange(-n/2., n/2.)**2./(2*sigma**2))/np.sqrt(2*sigma**2*np.pi)
    c[c < 1e-6] = 0
    b = np.convolve(c, x0) + 0.1*np.random.randn(2*n-1)

    x = cvx.Variable(n)
    f = cvx.sum_squares(cvx.conv(c, x) - b)
    return cvx.Problem(cvx.Minimize(f), [x >= 0])

PROBLEMS = [
    nn_deconv
]
SIZES = [100, 1000, 10000]


for prob_func in PROBLEMS:
    for n in SIZES:
        print prob_func.__name__
        print "num_vars:", n

        np.random.seed(0)
        prob = prob_func(n)

        if len(sys.argv) == 2 and sys.argv[1] == "scs":
            t0 = time.time()
            prob.solve(solver=cvx.SCS, verbose=True, gpu=True)
            print "gpu_solve_time: %.2f secs" % (time.time() - t0)

            t0 = time.time()
            prob.solve(solver=cvx.SCS, verbose=True, gpu=False)
            print "cpu_solve_time: %.2f secs" % (time.time() - t0)

            t0 = time.time()
            prob.get_problem_data(cvx.SCS)
            print "get_problem_data_time: %.2f secs" % (time.time() - t0)

        else:
            from cvxflow.problem import TensorProblem
            from cvxflow import scs_tf

            t0 = time.time()
            t_prob = TensorProblem(prob)
            print "problem_time:", time.time() - t0

            t0 = time.time()
            objective = scs_tf.solve(t_prob, max_iters=2500, gpu=True)
            print "gpu_solve_time: %.2f secs" % (time.time() - t0)
            print "objective: %.2e" % objective

            t0 = time.time()
            objective = scs_tf.solve(t_prob, max_iters=2500, gpu=False)
            print "cpu_solve_time: %.2f secs" % (time.time() - t0)
            print "objective: %.2e" % objective
