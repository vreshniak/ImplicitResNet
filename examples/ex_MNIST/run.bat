@echo off


FOR %%m IN (train test) DO (
	python ex_MNIST.py --name plain --mode %%m --theta 0.0 --adiv 0.0 --piters 0
	python ex_MNIST.py --name 1Lip  --mode %%m --theta 0.0 --adiv 0.0 --piters 1
)

FOR %%t IN (0.00 0.25 0.50 0.75 1.00) DO (
	FOR %%m IN (train test) DO (
		python ex_MNIST.py --name theta_%%t --mode %%m --theta %%t --stablim 0.2 1.2 1.5 --learn_scales --learn_shift
	)
)

python make_table.py
pdflatex results.tex
del *.aux *.log *.bak
