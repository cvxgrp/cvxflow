"""Examples of solving convex problems with TensorFlow."""

import numpy as np
import cvxpy as cvx
import time

from cvxflow.problem import TensorProblem
from cvxflow import scs_tf

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
SIZES = [1000] #, 1000] #10000]


for prob_func in PROBLEMS:
    for n in SIZES:
        print prob_func.__name__
        print "num_vars:", n

        np.random.seed(0)
        prob = prob_func(n)
        A = prob.get_problem_data(cvx.SCS)["A"]
        print "num_constraints:", A.shape[0]
        print "nnz:", A.nnz

        print "SCS"
        t0 = time.time()
        prob.solve(solver=cvx.SCS, verbose=True)
        print "solve_time: %.2f secs" % (time.time() - t0)
        print "objective: %.2e" % prob.objective.value

        print "tensorflow"
        t0 = time.time()
        t_prob = TensorProblem(prob)
        print "create_time:", time.time() - t0
        print "objective: %.2e" % scs_tf.solve(t_prob, max_iters=2500)
