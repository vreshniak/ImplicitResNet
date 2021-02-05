#!/bin/bash

for datasteps in 10 4 2; do
	for theta in 0.00 0.50 1.00; do
		for mode in train test; do
			python ex_stiff.py --name datasteps_"$datasteps"_theta_"$theta" --mode $mode --theta $theta --datasteps $datasteps
		done
	done
done

pdflatex results.tex
pdflatex results.tex
rm *.aux *.log *.bak
# open results.pdf
