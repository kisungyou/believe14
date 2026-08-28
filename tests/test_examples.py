"""Coverage and execution checks for the 30 method cards."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from believe14.api import Capability
from believe14.registry import list_estimators

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"

EXPECTED_CARD_FILES = {
    "PCA": "pca.md",
    "GaussianRandomProjection": "gaussian_random_projection.md",
    "FactorAnalysis": "factor_analysis.md",
    "ProbabilisticPCA": "probabilistic_pca.md",
    "FastICA": "fast_ica.md",
    "SparsePCA": "sparse_pca.md",
    "LinearDiscriminantAnalysis": "linear_discriminant_analysis.md",
    "CanonicalCorrelationAnalysis": "canonical_correlation_analysis.md",
    "PLSRegression": "pls_regression.md",
    "SlicedInverseRegression": "sliced_inverse_regression.md",
    "SlicedAverageVarianceEstimation": "sliced_average_variance_estimation.md",
    "FisherScore": "fisher_score.md",
    "ClassicalMDS": "classical_mds.md",
    "MetricMDS": "metric_mds.md",
    "SammonMapping": "sammon_mapping.md",
    "FastMap": "fast_map.md",
    "KernelPCA": "kernel_pca.md",
    "Isomap": "isomap.md",
    "LocallyLinearEmbedding": "locally_linear_embedding.md",
    "LaplacianEigenmaps": "laplacian_eigenmaps.md",
    "DiffusionMap": "diffusion_map.md",
    "LocalTangentSpaceAlignment": "local_tangent_space_alignment.md",
    "TSNE": "tsne.md",
    "PHATE": "phate.md",
    "CorrelationDimension": "correlation_dimension.md",
    "TwoNN": "two_nn.md",
    "LevinaBickelMLE": "levina_bickel_mle.md",
    "UStatisticDimension": "u_statistic_dimension.md",
    "MiNDML": "mi_ndml.md",
    "DANCo": "danco.md",
}

CODE_CELL = re.compile(r"```\{code-cell\} ipython3\n(.*?)```", re.DOTALL)


def _front_matter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match is not None, "method card is missing YAML front matter"
    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if line and not line.startswith(" ") and ":" in line:
            key, value = line.split(":", maxsplit=1)
            values[key] = value.strip()
    return values


def test_method_card_inventory_matches_registry_exactly() -> None:
    registry = {info.name: info for info in list_estimators()}
    assert len(registry) == 30
    assert set(EXPECTED_CARD_FILES) == set(registry)
    assert {path.name for path in EXAMPLES.glob("*.md")} == set(
        EXPECTED_CARD_FILES.values()
    )

    for name, filename in EXPECTED_CARD_FILES.items():
        metadata = _front_matter((EXAMPLES / filename).read_text(encoding="utf-8"))
        assert metadata["believe14_estimator"] == name
        assert metadata["believe14_family"] == registry[name].family
        assert metadata["kernelspec"]
        assert metadata["jupytext"] == ""


@pytest.mark.parametrize(
    ("name", "filename"), EXPECTED_CARD_FILES.items(), ids=EXPECTED_CARD_FILES
)
def test_method_card_contract(name: str, filename: str) -> None:
    info = next(item for item in list_estimators() if item.name == name)
    text = (EXAMPLES / filename).read_text(encoding="utf-8")
    cells = CODE_CELL.findall(text)

    assert len(cells) == 1
    assert f"# {name}" in text
    assert "\nUse " in text
    assert "diagnostic" in text.lower()
    assert "Primary reference: {cite:p}`" in text
    assert "shape" in cells[0] or info.family == "estimation"
    assert "http:" not in cells[0] and "https:" not in cells[0]
    assert "requests" not in cells[0] and "urlopen" not in cells[0]

    if Capability.TRANSFORM in info.capabilities:
        assert ".transform(" in cells[0]
    elif info.family == "nonlinear":
        assert 'not hasattr(model, "transform")' in cells[0]

    if Capability.STOCHASTIC in info.capabilities:
        assert "random_state=14" in cells[0]
        assert "replay" in cells[0]


@pytest.mark.parametrize(
    ("name", "filename"), EXPECTED_CARD_FILES.items(), ids=EXPECTED_CARD_FILES
)
def test_method_card_executes(
    name: str, filename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the same code cell that MyST-NB evaluates during the docs build."""

    monkeypatch.syspath_prepend(str(EXAMPLES))
    text = (EXAMPLES / filename).read_text(encoding="utf-8")
    (cell,) = CODE_CELL.findall(text)
    namespace = {"__name__": f"believe14_method_card_{name}"}
    exec(compile(cell, filename, "exec"), namespace)
