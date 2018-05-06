#!/bin/bash -eu

export PYTHONPATH=..

python="python"

# regularized least squares

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
