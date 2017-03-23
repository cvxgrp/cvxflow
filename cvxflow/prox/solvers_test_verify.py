
import numpy as np
import cvxpy as cvx

def lasso():
    A = np.array([[1,2],[3,4],[5,6]])
    b = np.array([7,8,9])
    x = cvx.Variable(2)
    f = 0.5*cvx.sum_squares(A*x - b)
    g = cvx.norm(x, 1)
    return x, cvx.Problem(cvx.Minimize(f + g))

def least_abs_dev():
    A = np.array([[1,1],[1,2],[1,3]])
    b = np.array([1,2,10])
    x = cvx.Variable(2)
    f = cvx.norm(A*x - b, 1)
    return x, cvx.Problem(cvx.Minimize(f))

def quantile_regression():
    A = np.array([[1,1],[1,2],[1,3],[1,4]])
    b = np.array([1,2,10,20])
    x = cvx.Variable(2)
    z = A*x - b
    tau = 0.2
    f = cvx.sum_entries(
        -tau*cvx.min_elemwise(z, 0) + (1-tau)*cvx.max_elemwise(z, 0))
    f += 1e-2*cvx.sum_squares(x)
    return x, cvx.Problem(cvx.Minimize(f))

def multiple_quantile_regression():
    A = np.array([[1.,1.],[1.,2.],[1.,3.],[1.,4.]])
    b = np.array([1.,2.,10.,20.])
    X = cvx.Variable(2,3)
    tau = np.array([0.2, 0.5, 0.8])

    Z = A*X - np.tile(b[:,np.newaxis],(1,3))
    f = cvx.sum_entries(
        cvx.mul_elemwise(np.tile(-tau[np.newaxis], (4,1)), cvx.min_elemwise(Z, 0)) +
        cvx.mul_elemwise(np.tile(1-tau[np.newaxis], (4,1)), cvx.max_elemwise(Z, 0)))
    f += 1e-4/2*cvx.sum_squares(X)
    return X, cvx.Problem(cvx.Minimize(f))


PROBLEMS = [lasso, least_abs_dev, quantile_regression, multiple_quantile_regression]

if __name__ == "__main__":
    for prob in PROBLEMS:
        x, cvx_prob = prob()
        cvx_prob.solve()
        print "%s:" % prob.__name__
        print cvx_prob.objective.value
        print repr(x.value)
