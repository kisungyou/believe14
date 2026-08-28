"""Read canonical release names from ``pyproject.toml``.

The release workflows call this module instead of embedding a version or archive
name in YAML.  It deliberately uses only the Python standard library so it can
run before project dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

_FINAL_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


@dataclass(frozen=True)
class ReleaseInfo:
    """Canonical project and artifact identifiers."""

    name: str
    version: str
    tag: str
    wheel: str
    sdist: str
    repository: str


def project_root() -> Path:
    """Return the repository root containing this script."""

    return Path(__file__).resolve().parents[1]


def load_release_info(root: Path | None = None) -> ReleaseInfo:
    """Load and validate release identifiers from project metadata."""

    base = project_root() if root is None else root.resolve()
    document = tomllib.loads((base / "pyproject.toml").read_text(encoding="utf-8"))
    project = document["project"]
    name = str(project["name"])
    version = str(project["version"])
    if _FINAL_VERSION.fullmatch(version) is None:
        raise ValueError(
            "The public release version must be a final numeric X.Y.Z value; "
            f"received {version!r}."
        )
    wheel_distribution = re.sub(r"[-_.]+", "_", name)
    sdist_distribution = re.sub(r"[-_.]+", "_", name)
    urls = project.get("urls", {})
    repository = str(urls.get("Repository", ""))
    return ReleaseInfo(
        name=name,
        version=version,
        tag=f"v{version}",
        wheel=f"{wheel_distribution}-{version}-py3-none-any.whl",
        sdist=f"{sdist_distribution}-{version}.tar.gz",
        repository=repository,
    )


def _write_github_output(path: Path, info: ReleaseInfo) -> None:
    values = asdict(info)
    with path.open("a", encoding="utf-8") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field",
        choices=("name", "version", "tag", "wheel", "sdist", "repository"),
    )
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    info = load_release_info()
    if arguments.github_output is not None:
        _write_github_output(arguments.github_output, info)
    if arguments.field is not None:
        print(getattr(info, arguments.field))
    elif arguments.github_output is None:
        print(json.dumps(asdict(info), sort_keys=True))


if __name__ == "__main__":
    main()
