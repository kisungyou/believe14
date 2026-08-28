"""Smoke-test an installed artifact without importing the source checkout."""

from __future__ import annotations

import argparse
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

import believe14
from believe14.estimation import TwoNN
from believe14.linear import PCA
from believe14.nonlinear import ClassicalMDS
from believe14.registry import list_estimators


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    module_path = Path(believe14.__file__).resolve()
    _require(
        not module_path.is_relative_to(project_root),
        f"Smoke test imported the source checkout: {module_path}",
    )
    _require(
        believe14.__version__ == arguments.expected_version,
        "Installed version is "
        f"{believe14.__version__}, not {arguments.expected_version}.",
    )

    installed = distribution("believe14")
    installed_files = {str(path) for path in installed.files or ()}
    _require("believe14/py.typed" in installed_files, "Artifact is missing py.typed.")
    _require(
        any(path.endswith("dist-info/licenses/LICENSE") for path in installed_files),
        "Artifact is missing its MIT license file.",
    )

    infos = list_estimators()
    _require(len(infos) == 30, "Artifact does not expose exactly 30 estimators.")
    _require(
        {info.family for info in infos} == {"linear", "nonlinear", "estimation"},
        "Artifact family inventory is incomplete.",
    )
    for info in infos:
        _require(
            info.estimator.__name__ == info.name,
            f"Registry import mismatch for {info.name}.",
        )

    rng = np.random.default_rng(1701)
    X = rng.normal(size=(64, 5))
    fitted = (
        PCA(2).fit(X),
        ClassicalMDS(2).fit(X),
        TwoNN().fit(X),
    )
    _require(
        np.all(np.isfinite(fitted[0].transform(X))),
        "Installed PCA produced a non-finite transform.",
    )
    _require(
        np.all(np.isfinite(fitted[1].embedding_)),
        "Installed ClassicalMDS produced a non-finite embedding.",
    )
    _require(
        np.isfinite(fitted[2].dimension_),
        "Installed TwoNN produced a non-finite estimate.",
    )
    for estimator in fitted:
        _require(
            estimator.diagnostics_.converged,
            f"Installed {type(estimator).__name__} did not converge.",
        )


if __name__ == "__main__":
    main()
