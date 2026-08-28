"""Verify release metadata, Git provenance, and tag invariants."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

if __package__:
    from .release_info import ReleaseInfo, load_release_info, project_root
else:  # pragma: no cover - direct script execution
    from release_info import ReleaseInfo, load_release_info, project_root


class ReleaseVerificationError(RuntimeError):
    """Raised when a release invariant is not satisfied."""


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ReleaseVerificationError(
            f"git {' '.join(arguments)} failed: {detail or 'unknown error'}"
        )
    return result.stdout.strip()


def _metadata_versions(root: Path) -> dict[str, str]:
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r"(?m)^version:\s*[\"']?([^\s\"']+)", citation)
    if citation_match is None:
        raise ReleaseVerificationError("CITATION.cff has no version field.")

    initializer = (root / "src/believe14/__init__.py").read_text(encoding="utf-8")
    fallback_match = re.search(
        r"(?m)^\s*__version__\s*=\s*[\"\']([^\"\']+)[\"\']", initializer
    )
    if fallback_match is None:
        raise ReleaseVerificationError(
            "src/believe14/__init__.py has no source-tree version fallback."
        )
    return {
        "CITATION.cff": citation_match.group(1),
        "src/believe14/__init__.py": fallback_match.group(1),
    }


def verify_metadata(root: Path, info: ReleaseInfo) -> None:
    """Verify every checked-in version declaration agrees with pyproject."""

    mismatches = {
        source: value
        for source, value in _metadata_versions(root).items()
        if value != info.version
    }
    if mismatches:
        raise ReleaseVerificationError(
            f"Version declarations disagree with {info.version}: {mismatches}"
        )

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(rf"(?m)^## \[{re.escape(info.version)}\](?:\s|$)")
    legacy_heading = re.compile(rf"(?m)^## {re.escape(info.version)}(?:\s|$)")
    if heading.search(changelog) is None and legacy_heading.search(changelog) is None:
        raise ReleaseVerificationError(
            f"CHANGELOG.md has no release heading for {info.version}."
        )

    if not info.repository:
        raise ReleaseVerificationError("Project metadata has no Repository URL.")


def _normalized_remote(value: str) -> str:
    normalized = value.strip().removesuffix(".git").removesuffix("/")
    normalized = normalized.replace("git@github.com:", "https://github.com/")
    normalized = normalized.replace("ssh://git@github.com/", "https://github.com/")
    if normalized.startswith(("http://", "https://")):
        parsed = urlsplit(normalized)
        if parsed.hostname is not None:
            normalized = f"https://{parsed.hostname.lower()}{parsed.path}"
    return normalized


def verify_repository(
    root: Path,
    info: ReleaseInfo,
    *,
    check_clean: bool,
    check_remote: bool,
    require_tag: bool,
    expected_tag: str | None,
) -> None:
    """Verify Git state needed for reproducible release evidence."""

    _git(root, "rev-parse", "--verify", "HEAD")
    _git(root, "diff", "--check")

    forbidden_prefixes = (
        "build/",
        "dist/",
        "docs/_build/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
    )
    tracked = _git(root, "ls-files").splitlines()
    forbidden = sorted(path for path in tracked if path.startswith(forbidden_prefixes))
    if forbidden:
        raise ReleaseVerificationError(
            f"Generated files must not be tracked for a release: {forbidden}"
        )

    if check_clean:
        status = _git(root, "status", "--porcelain", "--untracked-files=all")
        if status:
            raise ReleaseVerificationError(
                "Release evidence requires a clean source tree.\n" + status
            )

    if check_remote:
        actual = _normalized_remote(_git(root, "remote", "get-url", "origin"))
        expected = _normalized_remote(info.repository)
        if actual != expected:
            raise ReleaseVerificationError(
                f"origin points to {actual!r}; expected {expected!r}."
            )

    if not require_tag:
        return
    tag = info.tag if expected_tag is None else expected_tag
    if tag != info.tag:
        raise ReleaseVerificationError(
            f"Release tag {tag!r} does not match metadata tag {info.tag!r}."
        )
    tag_type = _git(root, "cat-file", "-t", f"refs/tags/{tag}")
    if tag_type != "tag":
        raise ReleaseVerificationError(
            f"{tag} must be an annotated tag; Git reports object type {tag_type!r}."
        )
    head = _git(root, "rev-parse", "HEAD")
    tagged = _git(root, "rev-parse", f"refs/tags/{tag}^{{commit}}")
    if tagged != head:
        raise ReleaseVerificationError(
            f"{tag} resolves to {tagged}, but the checked-out commit is {head}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-clean", action="store_true")
    parser.add_argument("--check-remote", action="store_true")
    parser.add_argument("--require-tag", action="store_true")
    parser.add_argument("--expected-tag")
    parser.add_argument("--metadata-only", action="store_true")
    arguments = parser.parse_args()

    root = project_root()
    info = load_release_info(root)
    verify_metadata(root, info)
    if not arguments.metadata_only:
        verify_repository(
            root,
            info,
            check_clean=arguments.check_clean,
            check_remote=arguments.check_remote,
            require_tag=arguments.require_tag,
            expected_tag=arguments.expected_tag,
        )
    if arguments.metadata_only:
        print(f"Release metadata is valid for {info.tag}.")
    else:
        print(f"Release metadata and provenance are valid for {info.tag}.")


if __name__ == "__main__":
    main()
