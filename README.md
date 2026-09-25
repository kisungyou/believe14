# believe14

`believe14` is a paper-first Python library for dimensionality reduction and
intrinsic-dimension estimation. Version 0.1.0 contains 30 independently
implemented methods organized into three public families:
`believe14.linear`, `believe14.nonlinear`, and `believe14.estimation`.

`UStatisticDimension` is experimental: its reference implementation has not met
the prespecified scientific accuracy threshold, and no supported accuracy regime
is established. The other 29 methods carry validated status for their recorded
tests and documented scope; this is not a general accuracy or maturity guarantee.
The [validation evidence](docs/validation/index.md) explains these statuses.

The package follows the scikit-learn estimator protocol while keeping its
scientific implementations independent. NumPy and SciPy provide numerical
primitives; scikit-learn provides estimator interoperability and validation.

## Installation

```bash
python -m pip install believe14
```

Python 3.12 or newer is required.

## Example

```python
from believe14.linear import PCA

model = PCA(n_components=2)
embedding = model.fit_transform(X)
```

Each method has a validation ledger under `docs/validation/` recording its
normative equations, numerical conventions, and independent evidence.

## Documentation and method examples

The documentation contains an executable method card for every public class,
six task-oriented guides, and a registry-generated coverage table. Examples use
small deterministic datasets and run without network access. They distinguish
inductive reducers with a justified `transform` from transductive embeddings
that intentionally expose no out-of-sample operation.

Build the examples from the packaged wheel, rather than the source checkout:

```bash
make examples
```

## Development and validation

Create the locked development environment and add Twine, then use the same
commands as continuous integration. The local build uses uv's offline artifact
cache so documentation examples never fetch data or packages while executing.

```bash
uv sync --extra dev --extra docs
uv pip install twine
make quality
make test
make package-check
```

`make release-check` adds executable documentation, research checks, clean-tree
and repository provenance checks, installed wheel/sdist tests, and the complete
scientific release audit. The supported release matrix is Python 3.12–3.14 on
Linux, macOS, and Windows. See [RELEASING.md](RELEASING.md) for the trusted
TestPyPI, PyPI, GitHub Release, and documentation deployment process.

The original U-statistic accuracy failure remains recorded. The revised release
protocol requires a separate 500-replicate certificate for each original
scenario, with simultaneous 95% upper RMSE bounds below the unchanged limit,
as well as all other scientific checks. See the release guide for the explicit
distinction between historical panel outcomes and the current decision.

## Release trajectory

- **0.1.1:** correctness, packaging, and documentation fixes only.
- **0.2.0:** supervised and feature-selection expansion.
- **0.3.0:** nonlinear and landmark-method expansion.
- **0.4.0:** intrinsic-dimension estimation expansion.

New methods must land as complete vertical slices: paper ledger, literal oracle,
implementation, normal tests, method card, and guide integration. Performance
work is profiling-led, and the readable NumPy/SciPy path remains the permanent
reference implementation.
