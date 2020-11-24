#!/bin/bash

if [ $# -eq 0 ]; then
	mode="train"
else
	mode=$1
	# theta=$2
fi



function run {
	python ex_Lotka.py   \
		--mode     $mode \
		\
		--seed     10    \
		\
		--theta    $1    \
		--tol      1.e-6 \
		--T        10    \
		--steps    50    \
		\
		--codim    0     \
		--width    20    \
		--depth    4     \
		--sigma    relu  \
		\
		--scales   learn \
		--piters   0     \
		\
		--init     rnd   \
		--epochs   3000  \
		--lr       1.e-3 \
		--datasize 100   \
		--datasteps 50   \
		--batch     -1   \
		\
		--aTV      -1    \
		--mciters  1
}



for theta in 0.0 0.25 0.5 0.75 1.0; do
	run $theta
done