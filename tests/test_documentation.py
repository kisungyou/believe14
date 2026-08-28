from __future__ import annotations

import re
from pathlib import Path

from believe14.registry import list_estimators

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GUIDES = (
    "choosing-a-method",
    "linear-latent-representations",
    "supervised-and-paired-reductions",
    "distance-and-graph-embeddings",
    "stress-and-stochastic-embeddings",
    "intrinsic-dimension",
)
FRONT_MATTER = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)
FIELD = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_-]*):\s*(?P<value>.*?)\s*$")
CODE_CELL = re.compile(r"```\{code-cell\}\s+ipython3\s*\n(?P<body>.*?)\n```", re.DOTALL)


def _metadata(path: Path) -> dict[str, str]:
    match = FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    assert match is not None, f"Missing front matter: {path}"
    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        parsed = FIELD.match(line)
        if parsed is not None:
            fields[parsed.group("key")] = parsed.group("value").strip("'\"")
    return fields


def test_six_executable_guides_are_in_the_site_navigation() -> None:
    guide_paths = sorted((DOCS / "guides").glob("*.md"))
    assert {path.stem for path in guide_paths} == set(GUIDES)
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    for slug in GUIDES:
        path = DOCS / "guides" / f"{slug}.md"
        text = path.read_text(encoding="utf-8")
        assert f"guides/{slug}" in index
        assert "jupytext:" in text and "kernelspec:" in text
        assert CODE_CELL.search(text), f"Guide is not executable: {path}"
    assert "examples/*" in index
    assert "references" in index


def test_method_card_metadata_exactly_covers_the_registry() -> None:
    cards = sorted((DOCS / "examples").glob("*.md"))
    assert len(cards) == 30
    observed: dict[str, tuple[str, Path]] = {}
    for path in cards:
        metadata = _metadata(path)
        name = metadata["believe14_estimator"]
        assert name not in observed, f"Duplicate method card for {name}"
        observed[name] = (metadata["believe14_family"], path)
        text = path.read_text(encoding="utf-8")
        assert "jupytext:" in text and "kernelspec:" in text
        assert CODE_CELL.search(text), f"Method card is not executable: {path}"

    expected = {info.name: info.family for info in list_estimators()}
    assert {name: family for name, (family, _) in observed.items()} == expected


def test_executable_documentation_is_offline_and_deterministic() -> None:
    paths = sorted((DOCS / "examples").glob("*.md")) + sorted(
        (DOCS / "guides").glob("*.md")
    )
    forbidden = re.compile(
        r"\b(?:requests|urlopen|urlretrieve|fetch_openml|fetch_\w+|load_digits)\b|"
        r"https?://"
    )
    for path in paths:
        for match in CODE_CELL.finditer(path.read_text(encoding="utf-8")):
            assert forbidden.search(match.group("body")) is None, (
                f"Executable documentation must be offline: {path}"
            )


def test_every_documentation_citation_has_a_central_bibliography_entry() -> None:
    citation = re.compile(r"\{cite:[^}]+\}`([^`]+)`")
    cited: set[str] = set()
    for path in (
        *sorted((DOCS / "examples").glob("*.md")),
        *sorted((DOCS / "guides").glob("*.md")),
    ):
        for group in citation.findall(path.read_text(encoding="utf-8")):
            cited.update(key.strip() for key in group.split(","))
    bibliography = (DOCS / "references.bib").read_text(encoding="utf-8")
    defined = set(re.findall(r"(?m)^@\w+\{([^,]+),", bibliography))
    assert cited
    assert cited <= defined


def test_release_docs_can_require_an_installed_wheel() -> None:
    config = (DOCS / "conf.py").read_text(encoding="utf-8")
    assert 'nb_execution_mode = "force"' in config
    assert "nb_execution_raise_on_error = True" in config
    assert 'os.environ.get("BELIEVE14_DOCS_REQUIRE_WHEEL")' in config
    assert 'PROJECT_ROOT / "src"' in config
    assert "PYTHONPATH" not in config
