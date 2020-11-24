#!/bin/bash

if [ $# -eq 0 ]; then
	mode="train"
else
	mode=$1
	# datasteps=$2
fi



function run {
	python ex_stiff.py   \
		--mode     $mode \
		\
		--seed     10    \
		\
		--theta    $1    \
		--tol      1.e-6 \
		--T        2     \
		--steps    20    \
		\
		--codim    0     \
		--width    4     \
		--depth    2     \
		--sigma    relu  \
		\
		--scales   learn \
		--piters   1     \
		--eigs   -25 -15 \
		\
		--init     rnd   \
		--epochs   50    \
		--lr       1.e-2 \
		--datasize 100   \
		--datasteps $2   \
		--batch     1    \
		\
		--aTV      1.e-1 \
		--mciters  1
}


for datasteps in 1 2 4 10 20; do
	for theta in 0.0 0.25 0.5 0.75 1.0; do
		run $theta $datasteps
	done
done