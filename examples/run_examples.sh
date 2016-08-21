#!/bin/bash -eu

export PYTHONPATH=..

python="/usr/bin/time -v python"

$python linear.py tensorflow dense_matrix
$python linear.py tensorflow sparse_matrix
$python linear.py tensorflow convolution
$python linear.py spsolve dense_matrix
$python linear.py spsolve sparse_matrix
$python linear.py spsolve convolution

$python convex.py tensorflow 100
$python convex.py tensorflow 1000
$python convex.py tensorflow 10000
$python convex.py scs 100
$python convex.py scs 1000
$python convex.py scs 10000
