PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
UV ?= uv
UV_CACHE_ENV = $(if $(UV_CACHE_DIR),UV_CACHE_DIR=$(UV_CACHE_DIR),)
DIST_DIR ?= dist
DOCS_BUILD_DIR ?= docs/_build/html
AUDIT_OUTPUT ?= build/release-audit.json

VERSION := $(shell $(PYTHON) -m scripts.release_info --field version)
TAG := $(shell $(PYTHON) -m scripts.release_info --field tag)
WHEEL := $(DIST_DIR)/$(shell $(PYTHON) -m scripts.release_info --field wheel)
SDIST := $(DIST_DIR)/$(shell $(PYTHON) -m scripts.release_info --field sdist)
ARTIFACT_ENV := $(PYTHON) -m scripts.run_in_artifact_env --installer uv

.PHONY: help quality test research build package-check package-check-built \
	docs docs-built examples examples-or-docs metadata-check verify-release \
	verify-tag release-audit release-audit-built release-check

help:
	@echo "believe14 $(VERSION) development commands"
	@echo "  quality          formatting, lint, and strict typing checks"
	@echo "  test             deterministic tests with branch coverage"
	@echo "  examples         executable documentation built from the wheel"
	@echo "  package-check    wheel/sdist metadata and installed smoke checks"
	@echo "  release-check    every local release gate"

quality:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest --cov=believe14 --cov-report=term-missing --cov-fail-under=90

research:
	$(PYTHON) -m pytest -m research

build:
	$(UV_CACHE_ENV) $(UV) build --offline --no-create-gitignore --out-dir $(DIST_DIR)

package-check: build package-check-built

package-check-built:
	$(PYTHON) -m twine check --strict $(WHEEL) $(SDIST)
	$(PYTHON) -m scripts.check_distribution --dist-dir $(DIST_DIR)
	$(PYTHON) -m scripts.artifact_hashes write --manifest $(DIST_DIR)/SHA256SUMS $(WHEEL) $(SDIST)
	$(ARTIFACT_ENV) --artifact $(WHEEL) -- {python} -I scripts/installed_smoke.py --expected-version $(VERSION)
	$(ARTIFACT_ENV) --artifact $(SDIST) -- {python} -I scripts/installed_smoke.py --expected-version $(VERSION)

docs: build docs-built

docs-built:
	BELIEVE14_DOCS_REQUIRE_WHEEL=1 $(ARTIFACT_ENV) --artifact $(WHEEL) -- {python} -I -m sphinx -W -b html docs $(DOCS_BUILD_DIR)
	$(ARTIFACT_ENV) --artifact $(WHEEL) -- {python} -I tools/check_docs_html.py $(DOCS_BUILD_DIR)

examples: docs

examples-or-docs: docs

metadata-check:
	$(PYTHON) -m scripts.verify_release --metadata-only

verify-release:
	$(PYTHON) -m scripts.verify_release --check-clean --check-remote

verify-tag:
	$(PYTHON) -m scripts.verify_release --check-clean --check-remote --require-tag --expected-tag $(TAG)

release-audit: package-check release-audit-built

release-audit-built:
	$(PYTHON) -c "from pathlib import Path; Path('$(AUDIT_OUTPUT)').parent.mkdir(parents=True, exist_ok=True)"
	$(ARTIFACT_ENV) --artifact $(WHEEL) -- {python} -I -W error tools/release_audit.py --output $(AUDIT_OUTPUT) --artifact $(WHEEL) --artifact $(SDIST)

release-check:
	$(MAKE) verify-release
	$(MAKE) quality
	$(MAKE) test
	$(MAKE) research
	$(MAKE) build
	$(MAKE) package-check-built
	$(MAKE) docs-built
	$(MAKE) release-audit-built
