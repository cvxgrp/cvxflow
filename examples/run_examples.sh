#!/bin/bash -eu

export PYTHONPATH=..

python="/usr/bin/time -v python"

# regularized least squares
$python linear.py tensorflow dense_matrix
$python linear.py tensorflow sparse_matrix
$python linear.py tensorflow convolution
$python linear.py spsolve dense_matrix
$python linear.py spsolve sparse_matrix
$python linear.py spsolve convolution

# lasso
$python convex.py tensorflow lasso_dense  1000
$python convex.py tensorflow lasso_sparse 1000
$python convex.py tensorflow lasso_conv   1000
$python convex.py scs lasso_dense  1000
$python convex.py scs lasso_sparse 1000
$python convex.py scs lasso_conv   1000

# nonnegative deconvolution
$python convex.py tensorflow nn_deconv 100
$python convex.py tensorflow nn_deconv 1000
$python convex.py tensorflow nn_deconv 10000
$python convex.py scs nn_deconv 100
$python convex.py scs nn_deconv 1000
$python convex.py scs nn_deconv 10000
