PYTHON ?= python
CLI = $(PYTHON) -m src.cli

.PHONY: help bootstrap generate test notebooks-run notebooks-run-inplace notebooks-export clean-notebook-exports

help:
	@echo "Available targets:"
	@echo "  bootstrap              Create raw/checkpoint notebook datasets"
	@echo "  generate               Generate sample GAN synthetic data"
	@echo "  test                   Run pytest suite"
	@echo "  notebooks-run          Execute all notebooks into notebooks/.executed"
	@echo "  notebooks-run-inplace  Execute all notebooks in place"
	@echo "  notebooks-export       Convert all notebooks to Python scripts"
	@echo "  clean-notebook-exports Remove generated notebook export folders"

bootstrap:
	$(CLI) bootstrap --samples 1000 --random-state 42

generate:
	$(CLI) generate --method gan --samples 1000 --output data/synthetic/gan_out.csv --random-state 42

test:
	$(CLI) test --suite all

notebooks-run:
	mkdir -p notebooks/.executed
	$(CLI) run-notebook --all --output-dir notebooks/.executed

notebooks-run-inplace:
	$(CLI) run-notebook --all --inplace

notebooks-export:
	mkdir -p notebooks/python
	$(CLI) export-notebooks --output-dir notebooks/python

clean-notebook-exports:
	rm -rf notebooks/.executed notebooks/python
