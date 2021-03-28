#!/bin/bash

for augm in clean; do
	for data in 1000; do
		for T in 1 3; do
			for mode in train test; do
				python ex_MNIST.py --name "$augm"_data_"$data"_T_"$T"_plain --model 1 --datasize $data --mode $mode --T $T --theta 0.0 --adiv 0.0 --piters 0
				python ex_MNIST.py --name "$augm"_data_"$data"_T_"$T"_1Lip  --model 1 --datasize $data --mode $mode --T $T --theta 0.0 --adiv 0.0 --piters 1 --stablim 0.0 1.0 2.0
			done

			for adiv in 1.00; do
				for theta in 0.00 0.25 0.50 0.75 1.00; do
					for lim in -1.0 -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4; do
						for mode in  test; do
							python -W ignore ex_MNIST.py --name "$augm"_data_"$data"_T_"$T"_adiv_"$adiv"_theta_"$theta"_lim_"$lim" --model 1 --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --stablim $lim 1.0 2.0 --learn_shift
						done
					done
				done
			done
		done
	done
done

python make_table.py
pdflatex results.tex
rm *.aux *.log *.bak
# open results.pdf
