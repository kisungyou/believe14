# Development

Scientific changes to `believe14` advance as complete vertical slices: a paper
specification, an independent literal oracle, an implementation, numerical and
API tests, an executable method card, and validation evidence.

## Local checks

The repository exposes the same small command interface used by continuous
integration:

```console
make quality
make test
make examples
make package-check
make release-check
```

`make examples` builds the documentation against the packaged wheel rather
than the source checkout. `make release-check` adds research checks, clean-tree
and repository-provenance verification, installed-artifact tests, and the
complete scientific release audit.

## Adding or changing a method

Read {doc}`Contributing scientific methods <../contributing>` before changing
an estimator. It defines the required ledger fields, numerical implementation
policy, and the only accepted promotion sequence. Public status is withheld
when a formula is ambiguous, convergence is false, calibration fails, or an
out-of-sample rule lacks a normative source.

## Validation evidence

The {doc}`validation ledgers <../validation/index>` record each public method's
normative reference, formulation, numerical conventions, failure domain, and
independent evidence. The machine-readable ledger and Python registry must
agree exactly.

## Releasing

The repository's [release guide](https://github.com/kisungyou/believe14/blob/main/RELEASING.md)
documents clean annotated tags, immutable wheel and source artifacts, TestPyPI
rehearsal, PyPI publication, GitHub Releases, and documentation deployment.

```{toctree}
:hidden:
:maxdepth: 2

Contributing scientific methods <../contributing>
Validation evidence <../validation/index>
```
