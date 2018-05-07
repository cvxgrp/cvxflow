#!/bin/bash -eu

export PYTHONPATH=..

python="python"



# lasso
$python convex.py tensorflow lasso_dense  1000

