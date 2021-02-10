#!/bin/bash

pytest -rA test_solvers.py
pytest -W ignore test_spectral_norm.py
# python test_spectral_norm.py
