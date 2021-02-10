#!/bin/bash

for mode in train test; do
	python ex_MNIST.py --name plain --mode $m --theta 0.0 --adiv 0.0 --piters 0
	python ex_MNIST.py --name 1Lip  --mode $m --theta 0.0 --adiv 0.0 --piters 1
done

for mode in train test; do
	for theta in 0.10 0.25 0.50 0.75 1.00; do
		python ex_MNIST.py --name theta_"$theta" --mode $mode --theta $theta --stablim 0.3 1.2 1.5 --learn_scales --learn_shift
	done
done

python make_table.py
pdflatex results.tex
rm *.aux *.log *.bak
# open results.pdf
