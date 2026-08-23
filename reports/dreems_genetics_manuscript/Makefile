.PHONY: assets refresh-assets pdf-from-sources pdf check clean

refresh-assets:
	cd ../.. && uv run --no-sync python reports/dreems_genetics_manuscript/build_assets.py

assets: refresh-assets

pdf-from-sources: refresh-assets pdf

pdf:
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	cd ../.. && uv run --no-sync python reports/dreems_genetics_manuscript/build_assets.py --manifest-only

check: pdf
	cd ../.. && uv run --no-sync python reports/dreems_genetics_manuscript/check_manuscript.py

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
