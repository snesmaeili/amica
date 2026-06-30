# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build

.PHONY: help clean html linkcheck doctest

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) \
		-b html \
		-W \
		--keep-going \
		"$(SOURCEDIR)" \
		"$(BUILDDIR)/html" \
		$(SPHINXOPTS) $(O)

linkcheck:
	@$(SPHINXBUILD) \
		-b linkcheck \
		"$(SOURCEDIR)" \
		"$(BUILDDIR)/linkcheck" \
		$(SPHINXOPTS) $(O)

doctest:
	@$(SPHINXBUILD) \
		-b doctest \
		"$(SOURCEDIR)" \
		"$(BUILDDIR)/doctest" \
		$(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
