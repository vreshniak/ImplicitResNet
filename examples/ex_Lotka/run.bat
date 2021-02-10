@echo off

FOR %%t IN (0.00 0.50 1.00) DO (
	FOR %%m IN (train test) DO (
		python ex_Lotka.py --name theta_%%t --mode %%m --theta %%t
	)
)

pdflatex results.tex
pdflatex results.tex
del *.aux *.log *.bak
