@echo off

FOR %%d IN (10 4 2) DO (
	FOR %%t IN (0.00 0.50 1.00) DO (
		FOR %%m IN (train test) DO (
			python ex_stiff.py --name datasteps_%%d_theta_%%t --mode %%m --theta %%t --datasteps %%d
		)
	)
)

pdflatex results.tex
pdflatex results.tex
del *.aux *.log *.bak
