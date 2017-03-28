from setuptools import setup, find_packages

setup(
    name = "cvxflow",
    version = "0.1.3",
    author = "Matt Wytock and Steven Diamond",
    url = "http://github.com/cvxgrp/cvxflow",
    author_email = "mwytock@stanford.edu, diamond@stanford.edu",
    packages = find_packages(),
    install_requires=[
        "cvxpy>=0.4.8",
        "tensorflow>=1.0.0",
    ],
    test_suite = "cvxflow",
)
