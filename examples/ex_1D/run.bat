@echo off


FOR %%a IN (0.00 0.50 1.00 2.00 3.00) DO (
	FOR %%t IN (0.00 0.25 0.50 0.75 1.00) DO (
		FOR %%m IN (train test) DO (
			python ex_1D.py --name adiv_%%a_theta_%%t --mode %%m --theta %%t --adiv %%a
		)
	)
)

python extract_log_scalars.py
pdflatex results.tex
del *.aux *.log *.bak
