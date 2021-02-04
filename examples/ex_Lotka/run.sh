#!/bin/bash

for theta in 0.00 0.50 1.00; do
	for mode in train test; do
		python ex_Lotka.py --name theta_"$theta" --mode $mode --theta $theta
	done
done

pdflatex results.tex
pdflatex results.tex
rm *.aux *.log *.bak
# open results.pdf