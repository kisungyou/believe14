from __future__ import annotations

import inspect
import os
import re
import tomllib
from pathlib import Path

import pytest

import believe14
from believe14.api import Capability, FitDiagnostics
from believe14.registry import (
    get_estimator,
    list_estimators,
    public_estimator_names,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROOT / "docs" / "validation"

LEDGER_CONCEPTS = {
    "normative source": re.compile(
        r"primary source|authority|implementation follows|doi:|pearson \(1901\)",
        re.IGNORECASE,
    ),
    "frozen formulation": re.compile(
        r"definition|specification|statistic|model:|objective:|kernel:|"
        r"scatter definitions|probabilities:|norm component|frozen convention|"
        r"solver:|symmetric update",
        re.IGNORECASE,
    ),
    "numerical contract": re.compile(
        r"failure|reject|converg|stopping|constraint|degener|conventions|ties|"
        r"numerical",
        re.IGNORECASE,
    ),
    "output or out-of-sample contract": re.compile(
        r"transform|transductive|out.of.sample|selector",
        re.IGNORECASE,
    ),
    "complexity": re.compile(r"complexity|cost", re.IGNORECASE),
    "independent evidence": re.compile(r"evidence", re.IGNORECASE),
    "legacy divergence": re.compile(r"Rdimtools", re.IGNORECASE),
}
PLACEHOLDER = re.compile(
    r"(?im)\b(?:TODO|TBD|PLACEHOLDER)\b|^\s*-\s*(?:Primary reference|Equations|"
    r"Errata/supplement|Exact objective or statistic|Preprocessing and normalization|"
    r"Input and parameter domain|Degenerate inputs and failures|Solver, initialization,"
    r" and stopping|Randomness|Output equivalence|Out-of-sample behavior|Complexity|"
    r"Literal oracle|Independent comparator or paper fixture|Metamorphic properties|"
    r"Numerical stress cases|Empirical advertised regime):\s*$"
)


def test_static_version_is_numeric() -> None:
    assert believe14.__version__ == "0.1.0"
    assert "dev" not in believe14.__version__


def test_release_version_agrees_across_artifacts_and_tag() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    project_version = project["project"]["version"]
    assert (
        "/uv.lock" in project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    )
    with (VALIDATION_ROOT / "methods.toml").open("rb") as stream:
        manifest_version = tomllib.load(stream)["release"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r"(?m)^version:\s*([^\s#]+)\s*$", citation)
    assert citation_match is not None
    assert {
        believe14.__version__,
        project_version,
        manifest_version,
        citation_match.group(1),
    } == {"0.1.0"}

    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        assert os.environ.get("GITHUB_REF_NAME") == f"v{project_version}"


def test_registry_has_exact_release_inventory() -> None:
    infos = list_estimators()
    assert len(infos) == 30
    assert len(set(public_estimator_names())) == 30
    assert sum(info.family == "linear" for info in infos) == 12
    assert sum(info.family == "nonlinear" for info in infos) == 12
    assert sum(info.family == "estimation" for info in infos) == 6
    for info in infos:
        assert info.estimator.__module__.startswith(f"believe14.{info.family}")
        parameters = inspect.signature(info.estimator.__init__).parameters.values()
        assert all(
            parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
            for parameter in parameters
        )


def test_manifest_registry_exports_and_ledgers_agree() -> None:
    manifest_path = VALIDATION_ROOT / "methods.toml"
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)
    records = manifest["methods"]
    assert manifest["release"] == "0.1.0"
    assert len(records) == 30
    assert len({record["name"] for record in records}) == 30
    assert len({record["ledger"] for record in records}) == 30
    assert {record["name"] for record in records} == set(public_estimator_names())
    assert {record["state"] for record in records} == {"public"}
    expected_ledgers: set[Path] = set()
    for record in records:
        relative = Path(record["ledger"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert relative.parts[0] == record["family"]
        ledger = VALIDATION_ROOT / relative
        assert ledger.is_file(), f"Missing validation ledger: {ledger}"
        expected_ledgers.add(ledger)
        text = ledger.read_text(encoding="utf-8")
        assert text.startswith(f"# {record['name']} validation ledger\n")
        assert len(text) >= 500, f"Validation ledger is too small: {ledger}"
        assert re.search(r"(?im)^-\s+(?:\*\*)?status:(?:\*\*)?\s+validated\b", text), (
            f"Ledger does not have validated status: {ledger}"
        )
        assert not PLACEHOLDER.search(text), f"Ledger contains a placeholder: {ledger}"
        for concept, pattern in LEDGER_CONCEPTS.items():
            assert pattern.search(text), f"Ledger lacks {concept}: {ledger}"
        module = __import__(f"believe14.{record['family']}", fromlist=[record["name"]])
        assert record["name"] in module.__all__

    actual_ledgers = {
        path
        for family in ("linear", "nonlinear", "estimation")
        for path in (VALIDATION_ROOT / family).glob("*.md")
    }
    assert actual_ledgers == expected_ledgers


def test_fit_diagnostics_is_immutable() -> None:
    diagnostics = FitDiagnostics(solver="exact", converged=True)
    try:
        diagnostics.solver = "changed"  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:  # pragma: no cover
        raise AssertionError("FitDiagnostics must be immutable")


def test_registry_filters_and_lookups_are_explicit() -> None:
    linear = list_estimators(family="linear")
    assert len(linear) == 12
    supervised = list_estimators(supervision="supervised")
    assert supervised and all(info.supervision == "supervised" for info in supervised)
    stochastic = list_estimators(capability="stochastic")
    assert stochastic and all(
        Capability.STOCHASTIC in info.capabilities for info in stochastic
    )
    assert "PHATE" not in {info.name for info in stochastic}
    graph = list_estimators(approach="graph")
    assert {info.name for info in graph} == {
        "Isomap",
        "LocallyLinearEmbedding",
        "LaplacianEigenmaps",
        "DiffusionMap",
        "LocalTangentSpaceAlignment",
        "PHATE",
    }
    assert list_estimators(approach="does_not_exist") == ()
    assert get_estimator("PCA") is get_estimator("believe14.linear.PCA")
    assert get_estimator("FastMap").complexity == (
        "O(n^2 p + k n^2) time; O(n^2) memory"
    )
    with pytest.raises(KeyError, match="Unknown public estimator"):
        get_estimator("NotAnEstimator")
    with pytest.raises(ValueError):
        list_estimators(capability="not-a-capability")
