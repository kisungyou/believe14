# Releasing believe14

The public artifacts are built once from a clean annotated tag. The same wheel
and source archive are checked on all supported platforms, rehearsed on
TestPyPI, uploaded unchanged to PyPI, and attached to the GitHub Release.

## One-time repository setup

1. Create `kisungyou/believe14` on GitHub, set `main` as its default protected
   branch, and enable GitHub Actions as the Pages source.
2. Add GitHub environments named `testpypi`, `pypi`, and `github-pages`.
   Protection rules and required reviewers may be added to all three.
3. Register `.github/workflows/publish.yml` as a trusted publisher for the
   `testpypi` environment on TestPyPI and the `pypi` environment on PyPI. No API
   token or repository secret is used.
4. Require the CI workflow before merging to `main`. The workflow exercises
   Python 3.12–3.14 on Linux, macOS, and Windows and deploys documentation only
   after every required job passes.

## Prepare 0.1.0

Use the locked development environment containing the `dev` and `docs` extras
plus Twine. The local release command intentionally fails when the Git tree is
dirty or `origin` does not match the repository URL in `pyproject.toml`.

```bash
uv sync --extra dev --extra docs
uv pip install twine
make release-check
```

Review `dist/SHA256SUMS` and `build/release-audit.json`. Confirm that the release
gate in the audit is `passed: true`. Commit only source files; `dist/`, `build/`,
and generated documentation are ignored and must never be committed.

## Tag and publish

Create the release tag only from a clean, passing `main` commit. It must be an
annotated tag whose name exactly matches `v` plus the numeric project version.

```bash
git switch main
git pull --ff-only
make release-check
git tag -a v0.1.0 -m "believe14 0.1.0"
make verify-tag
git push origin main
git push origin v0.1.0
```

The tag starts the publish workflow. That workflow:

1. verifies the clean tag, metadata, examples, tests, and scientific audit;
2. builds exactly one wheel and one sdist and records their SHA-256 digests;
3. smoke-tests wheel and sdist installations on every supported Python/OS pair;
4. uploads those artifacts to TestPyPI through trusted publishing and installs
   the TestPyPI copy as a rehearsal;
5. uploads the original, checksum-verified artifacts to PyPI;
6. deploys documentation built against the wheel from the tagged commit; and
7. creates the GitHub Release with the same two files and `SHA256SUMS`.

Do not rebuild, edit, or upload artifacts manually between these stages. If a
stage fails before PyPI publication, fix the source and create a new version;
released filenames are immutable. A duplicate TestPyPI version is also treated
as a failure rather than silently skipped.

## After publication

Verify the PyPI project page, the GitHub Release checksums, and the deployed
documentation. Installation compatibility or correctness regressions go into
0.1.1; new methods belong to the planned 0.2.0 or later catalog expansions.
