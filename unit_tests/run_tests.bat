@echo off

pytest -rA test_solvers.py
pytest -W ignore test_spectral_norm.py
rem python test_spectral_norm.py
