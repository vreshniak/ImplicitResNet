#!/bin/bash

if [ $# -eq 0 ]; then
	mode="train"
else
	mode=$1
	# adiv=$2
fi


function run {
	python ex_1D.py      \
		--mode     $mode \
		\
		--seed     10    \
		\
		--theta    $1    \
		--tol      1.e-6 \
		--T        5     \
		\
		--codim    1     \
		--width    10    \
		--depth    3     \
		--sigma    gelu  \
		\
		--scales   equal \
		--piters   0     \
		\
		--init     rnd   \
		--epochs   3000  \
		--lr       1.e-3 \
		--datasize 20    \
		--batch    -1    \
		\
		--aTV      -1.0  \
		--adiv     $2    \
		--ajac     1.e-1 \
		--mciters  1
}



for adiv in 0.0 0.25 0.5 0.75 1.0; do
	for theta in 0.0 0.25 0.5 0.75 1.0; do
		run $theta $adiv
	done
done
