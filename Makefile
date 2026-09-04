.PHONY: help verify test spatial qtl-smoke qtl-batch

help:
	@echo "verify     check size, private paths, checksums, and syntax"
	@echo "test       run Python unit tests"
	@echo "spatial    reproduce per-frame vine coverage"
	@echo "qtl-smoke  run lab_b_mean with two permutations"
	@echo "qtl-batch  run all continuous traits with 1000 permutations"

verify:
	python scripts/verify_repository.py
	PYTHONPYCACHEPREFIX=/tmp/berry-color-qtl-pycache python -m compileall -q scripts tests
	@for script in scripts/*.sh scripts/qtl/*.sh; do bash -n "$$script"; done
	@for script in scripts/qtl/*.R; do Rscript -e "parse(file='$$script')" >/dev/null; done

test:
	PYTHONPYCACHEPREFIX=/tmp/berry-color-qtl-pycache pytest -q -p no:cacheprovider

spatial:
	bash scripts/run_spatial_mapping.sh

qtl-smoke:
	N_PERM=2 TRAITS=lab_b_mean OUT_DIR=results/generated/qtl/smoke_lab_b Rscript scripts/qtl/run_continuous_qtl.R

qtl-batch:
	N_PERM=1000 bash scripts/qtl/run_qtl_batch.sh
