#!/usr/bin/env bash
set -e

echo "Running util tests"
python test_util.py

echo "Running surrogate/integrator tests"
python test_surrogate.py
