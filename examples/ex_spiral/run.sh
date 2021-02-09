#!/bin/bash

for name in plain-10 1Lip-10 2Lip-10 2Lip-5-1Lip-5 10-0 5-5-0.00; do
	for theta in 0.00; do
		for mode in train test; do
			python ex_spiral.py --name $name --mode $mode --theta $theta
		done
	done
done

pdflatex results.tex
rm *.aux *.log *.bak