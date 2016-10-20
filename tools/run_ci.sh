#!/bin/bash -eux
#
# Script for running continuous integration tests

cd $(dirname "${BASH_SOURCE[0]}")/..

# Run Python 2.7 tests
python2 setup.py test

# Run Python 3.4 tests
python3 setup.py test
