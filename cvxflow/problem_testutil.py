
import cvxpy as cvx
import numpy as np

def least_squares():
    np.random.seed(0)
    m = 10
    n = 5
    A = np.random.randn(m,n)
    b = np.random.randn(m)
    x = cvx.Variable(n)
    return cvx.Problem(cvx.Minimize(cvx.sum_squares(A*x - b)))

def linear_program():
    np.random.seed(0)
    m = 5
    n = 10
    A = np.random.randn(m,n)
    b = A.dot(np.abs(np.random.randn(n)))
    c = np.random.rand(n)
    x = cvx.Variable(n)
    return cvx.Problem(cvx.Minimize(c.T*x), [A*x == b, x >= 0])

def nonnegative_deconvolution():
    np.random.seed(0)
    n = 10
    k = 3
    sigma = n/10.

    c = (np.exp(-np.arange(-n/2., n/2.)**2./(2*sigma**2))/
         np.sqrt(2*sigma**2*np.pi))
    x0 = np.zeros(n)
    x0[np.random.choice(n,k)] = np.random.randn(k)*sigma
    b = np.convolve(c, x0)
    b += np.random.randn(2*n-1)*np.linalg.norm(b)/np.sqrt(2*n-1)/20

    x = cvx.Variable(n)
    f = cvx.sum_squares(cvx.conv(c, x) - b)
    return cvx.Problem(cvx.Minimize(f), [x >= 0])

PROBLEMS = [
    nonnegative_deconvolution,
    least_squares,
    linear_program,
]
