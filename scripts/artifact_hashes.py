"""Write or verify a deterministic SHA-256 manifest for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(manifest: Path, artifacts: tuple[Path, ...]) -> None:
    """Write sorted digests, rejecting missing or duplicate artifact names."""

    if not artifacts:
        raise ValueError("At least one artifact is required.")
    names = [artifact.name for artifact in artifacts]
    if len(names) != len(set(names)):
        raise ValueError("Artifact basenames must be unique.")
    missing = [str(artifact) for artifact in artifacts if not artifact.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release artifacts: {missing}")
    lines = [f"{sha256(path)}  {path.name}" for path in sorted(artifacts)]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_manifest(manifest: Path, artifacts: tuple[Path, ...]) -> None:
    """Verify that paths exactly match a previously written manifest."""

    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or re_full_digest(digest) is None or not name:
            raise ValueError(f"Malformed checksum line: {line!r}")
        if name in expected:
            raise ValueError(f"Duplicate checksum entry: {name}")
        expected[name] = digest
    observed = {artifact.name: sha256(artifact) for artifact in artifacts}
    if observed != expected:
        raise ValueError(
            f"Artifact checksums changed; expected={expected}, observed={observed}"
        )


def re_full_digest(value: str) -> str | None:
    """Return a valid lowercase SHA-256 digest, otherwise ``None``."""

    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("write", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", required=True, type=Path)
        child.add_argument("artifacts", nargs="+", type=Path)
    arguments = parser.parse_args()
    artifacts = tuple(arguments.artifacts)
    if arguments.command == "write":
        write_manifest(arguments.manifest, artifacts)
    else:
        verify_manifest(arguments.manifest, artifacts)


if __name__ == "__main__":
    main()
