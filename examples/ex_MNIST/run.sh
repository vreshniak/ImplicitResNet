#!/bin/bash

# if [ $# -eq 0 ]; then
# 	mode="train"
# else
# 	mode=$1
# fi


function run {
	python ex_MNIST.py   \
		--mode     $1    \
		\
		--seed     10    \
		\
		--theta    $2    \
		--tol      1.e-6 \
		--T        2     \
		\
		--codim    7     \
		--depth    2     \
		--sigma    relu  \
		\
		--scales   learn \
		--piters   3     \
		--eigs    -2.5 1 \
		\
		--init     rnd   \
		--epochs   200   \
		--lr       1.e-2 \
		--datasize 1000  \
		--batch    200   \
		\
		--aTV      1.e-3 \
		--aresid   1     \
		--adiv     1.e-1 \
		--mciters  1
}


for mode in test; do
	for theta in 0.0 0.25 0.5 0.75 1.0; do
		run $mode $theta
	done
done


python make_table.py
pdflatex results.tex
rm *.aux *.log
# open results.pdf