"""Helpers for constructing public numerical diagnostics."""

from __future__ import annotations

from believe14.api import FitDiagnostics


def diagnostics(
    solver: str,
    *,
    converged: bool = True,
    n_iter: int | None = None,
    residual_norm: float | None = None,
    objective_value: float | None = None,
    numerical_rank: int | None = None,
    condition_estimate: float | None = None,
    warnings: tuple[str, ...] = (),
) -> FitDiagnostics:
    """Build an immutable diagnostics record."""

    return FitDiagnostics(
        solver=solver,
        converged=converged,
        n_iter=n_iter,
        residual_norm=residual_norm,
        objective_value=objective_value,
        numerical_rank=numerical_rank,
        condition_estimate=condition_estimate,
        warnings=warnings,
    )
