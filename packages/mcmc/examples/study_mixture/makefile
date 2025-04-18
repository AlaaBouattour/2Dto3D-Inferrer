# makefile

all: main_report.pdf

main_report.pdf: main_report.tex
	pdflatex main_report.tex
	pdflatex main_report.tex

# Pour produire main_report.tex, on a besoin du script + params.txt
main_report.tex: mcmc_mixture_study.py params.txt
	python3 mcmc_mixture_study.py

clean:
	rm -f *.aux *.log *.out
