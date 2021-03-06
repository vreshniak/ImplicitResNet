#!/bin/bash

for adiv in 0.00 0.50 1.00 2.00 3.00; do
	for theta in 0.00 0.25 0.50 0.75 1.00; do
		for mode in train test; do
			python ex_1D.py --name adiv_"$adiv"_theta_"$theta" --mode $mode --theta $theta --adiv $adiv
		done
	done
done


python extract_log_scalars.py
pdflatex results.tex
rm *.aux *.log *.bak
# open results.pdf
