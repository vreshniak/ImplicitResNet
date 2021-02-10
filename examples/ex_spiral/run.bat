@echo off

FOR %%n IN (plain-10, 1Lip-10, 2Lip-10, 2Lip-5-1Lip-5, 10-0, 5-5-0.00) DO (
	FOR %%t IN (0.00) DO (
		FOR %%m IN ( test) DO (
			python ex_spiral.py --name %%n --mode %%m --theta %%t
		)
	)
)
pdflatex results.tex
del *.aux *.log *.bak
