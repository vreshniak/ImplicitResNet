@echo off


rem FOR %%m IN (train test) DO (
rem 	python ex_MNIST.py --name model_1_plain --model 1 --mode %%m --theta 0.0 --adiv 0.0 --piters 0 --lr 1.e-2
rem 	rem python ex_MNIST.py --name adiv0 --mode %%m --theta 0.0 --adiv 0.0 --piters 1 --lr 1.e-2 --stablim -1.0 1.0 2.0 --learn_shift
rem )


FOR %%a IN (0.00) DO (
	FOR %%t IN (0.20 0.30 0.40 0.50 0.75 1.00) DO (
		FOR %%m IN (train test) DO (
			python ex_MNIST.py --name model_1_adiv_%%a_theta_%%t --model 1 --mode %%m --theta %%t --lr 1.e-2 --adiv %%a --piters 1 --stablim 0.0 1.0 2.0 --learn_shift
		)
	)
)
rem --adaugm 1.e-5
rem --stablim 0.1 1.0 5.0 --learn_shift
rem rem --stablim 0.2 1.0 2.0 --learn_shift --adiv 0.1
rem python make_table.py
rem pdflatex results.tex
rem del *.aux *.log *.bak
