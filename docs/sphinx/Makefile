# Minimal Sphinx makefile.
#
#   pip install -r requirements.txt
#   make html        -> _build/html/index.html
#   make linkcheck   -> report dead links
#   make strict      -> treat warnings as errors, as CI should

SPHINXBUILD ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html strict linkcheck clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo "Built $(BUILDDIR)/html/index.html"

strict:
	@$(SPHINXBUILD) -b html -W --keep-going -n "$(SOURCEDIR)" "$(BUILDDIR)/html"

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck"

clean:
	rm -rf "$(BUILDDIR)"
