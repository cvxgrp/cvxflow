#!/bin/bash -eu

tag=cvxgrp/cvxflow
docker build -t $tag .
docker push $tag
