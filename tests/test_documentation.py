from __future__ import annotations

import glob
import itertools
import posixpath
import re
from dataclasses import dataclass
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
TOCTREE = re.compile(r"(?ms)^```\{toctree\}\s*\n(?P<body>.*?)^```\s*$")
EXPLICIT_TOCTREE_ENTRY = re.compile(r"^(?P<label>.+?)\s*<(?P<target>[^>]+)>$")
PRIMARY_NAVIGATION = (
    ("Getting started", "getting_started/index"),
    ("Tutorials", "tutorials/index"),
    ("Methods", "methods"),
    ("API reference", "api"),
    ("Development", "development/index"),
)
METHOD_FAMILY_PAGES = {
    "linear": "methods/linear",
    "nonlinear": "methods/nonlinear",
    "estimation": "methods/estimation",
}


@dataclass(frozen=True)
class _NavEntry:
    label: str
    target: str


def _metadata(path: Path) -> dict[str, str]:
    match = FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    assert match is not None, f"Missing front matter: {path}"
    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        parsed = FIELD.match(line)
        if parsed is not None:
            fields[parsed.group("key")] = parsed.group("value").strip("'\"")
    return fields


def _document_path(docname: str) -> Path:
    for suffix in (".md", ".rst"):
        candidate = DOCS / f"{docname}{suffix}"
        if candidate.is_file():
            return candidate
    raise AssertionError(f"Navigation target does not exist: {docname}")


def _document_title(docname: str) -> str:
    path = _document_path(docname)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        heading = re.search(r"(?m)^#\s+(.+?)\s*$", text)
        assert heading is not None, f"Missing level-one heading: {path}"
        return heading.group(1)

    lines = text.splitlines()
    for title, underline in itertools.pairwise(lines):
        if title.strip() and re.fullmatch(r"=+", underline.strip()):
            return title.strip()
    raise AssertionError(f"Missing reStructuredText title: {path}")


def _normalize_target(source: str, target: str) -> str:
    target = target.split("#", maxsplit=1)[0].strip()
    rooted = target.startswith("/")
    target = target.removeprefix("/")
    for suffix in (".md", ".rst"):
        target = target.removesuffix(suffix)
    base = "" if rooted else posixpath.dirname(source)
    return posixpath.normpath(posixpath.join(base, target))


def _expand_target(source: str, target: str) -> tuple[str, ...]:
    normalized = _normalize_target(source, target)
    if not glob.has_magic(normalized):
        _document_path(normalized)
        return (normalized,)

    matches: set[str] = set()
    for suffix in (".md", ".rst"):
        pattern = str(DOCS / f"{normalized}{suffix}")
        for match in glob.glob(pattern):
            path = Path(match)
            if path.is_file() and not path.name.startswith("_"):
                matches.add(path.relative_to(DOCS).with_suffix("").as_posix())
    assert matches, f"Navigation glob did not match any documents: {target}"
    return tuple(sorted(matches))


def _toctree_entries(docname: str) -> tuple[_NavEntry, ...]:
    path = _document_path(docname)
    text = path.read_text(encoding="utf-8")
    entries: list[_NavEntry] = []
    for directive in TOCTREE.finditer(text):
        for raw_line in directive.group("body").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(":") or line.startswith("#"):
                continue
            explicit = EXPLICIT_TOCTREE_ENTRY.fullmatch(line)
            label = explicit.group("label").strip() if explicit else None
            target = explicit.group("target").strip() if explicit else line
            if target == "self" or re.match(r"^[a-z]+://", target):
                continue
            expanded = _expand_target(docname, target)
            assert label is None or len(expanded) == 1, (
                f"An explicit label cannot name a navigation glob: {path}"
            )
            entries.extend(
                _NavEntry(label or _document_title(item), item) for item in expanded
            )
    return tuple(entries)


def _navigation_graph() -> dict[str, tuple[_NavEntry, ...]]:
    graph: dict[str, tuple[_NavEntry, ...]] = {}
    pending = ["index"]
    while pending:
        docname = pending.pop()
        if docname in graph:
            continue
        entries = _toctree_entries(docname)
        graph[docname] = entries
        pending.extend(entry.target for entry in entries)
    return graph


def _descendants(graph: dict[str, tuple[_NavEntry, ...]], owner: str) -> set[str]:
    found: set[str] = set()
    pending = [owner]
    while pending:
        parent = pending.pop()
        for entry in graph.get(parent, ()):
            if entry.target not in found:
                found.add(entry.target)
                pending.append(entry.target)
    return found


def _assert_single_navigation_parent(
    graph: dict[str, tuple[_NavEntry, ...]], docnames: set[str]
) -> None:
    parent_counts = {
        docname: sum(
            entry.target == docname for entries in graph.values() for entry in entries
        )
        for docname in docnames
    }
    assert parent_counts == {docname: 1 for docname in docnames}


def test_primary_navigation_has_five_ordered_sections() -> None:
    root_entries = _toctree_entries("index")
    assert tuple((entry.label, entry.target) for entry in root_entries) == (
        PRIMARY_NAVIGATION
    )


def test_documentation_hierarchy_is_complete_and_canonical() -> None:
    graph = _navigation_graph()
    root_targets = {target for _, target in PRIMARY_NAVIGATION}
    assert root_targets <= graph.keys()

    assert {entry.target for entry in graph["getting_started/index"]} == {
        "scientific-contract"
    }

    guide_paths = sorted((DOCS / "guides").glob("*.md"))
    assert {path.stem for path in guide_paths} == set(GUIDES)
    guide_docnames = {f"guides/{slug}" for slug in GUIDES}
    tutorial_descendants = _descendants(graph, "tutorials/index")
    assert {entry.target for entry in graph["tutorials/index"]} == guide_docnames
    assert guide_docnames <= tutorial_descendants
    assert graph["tutorials/index"][0].target == "guides/choosing-a-method"
    for slug in GUIDES:
        path = DOCS / "guides" / f"{slug}.md"
        text = path.read_text(encoding="utf-8")
        assert "jupytext:" in text and "kernelspec:" in text
        assert CODE_CELL.search(text), f"Guide is not executable: {path}"

    family_targets = set(METHOD_FAMILY_PAGES.values())
    method_children = {entry.target for entry in graph["methods"]}
    assert method_children == family_targets | {"references"}
    card_docnames: set[str] = set()
    cards_by_family: dict[str, set[str]] = {
        family: set() for family in METHOD_FAMILY_PAGES
    }
    for path in sorted((DOCS / "examples").glob("*.md")):
        family = _metadata(path)["believe14_family"]
        docname = f"examples/{path.stem}"
        card_docnames.add(docname)
        cards_by_family[family].add(docname)
        assert docname in _descendants(graph, METHOD_FAMILY_PAGES[family])
        for other_family, owner in METHOD_FAMILY_PAGES.items():
            if other_family != family:
                assert docname not in _descendants(graph, owner)
    for family, owner in METHOD_FAMILY_PAGES.items():
        assert {entry.target for entry in graph[owner]} == cards_by_family[family]

    development_children = {entry.target for entry in graph["development/index"]}
    assert development_children == {"contributing", "validation/index"}
    validation_ledgers = {
        path.relative_to(DOCS).with_suffix("").as_posix()
        for path in (DOCS / "validation").glob("*/*.md")
    }
    assert {entry.target for entry in graph["validation/index"]} == validation_ledgers

    canonical_documents = (
        root_targets
        | {"scientific-contract", "references", "contributing", "validation/index"}
        | guide_docnames
        | family_targets
        | card_docnames
        | validation_ledgers
    )
    _assert_single_navigation_parent(graph, canonical_documents)


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
    quickstart = DOCS / "getting_started" / "index.md"
    paths = (
        quickstart,
        *sorted((DOCS / "examples").glob("*.md")),
        *sorted((DOCS / "guides").glob("*.md")),
    )
    forbidden = re.compile(
        r"\b(?:requests|urlopen|urlretrieve|fetch_openml|fetch_\w+|load_digits)\b|"
        r"https?://"
    )
    assert "jupytext:" in quickstart.read_text(encoding="utf-8")
    assert CODE_CELL.search(quickstart.read_text(encoding="utf-8"))
    for path in paths:
        for match in CODE_CELL.finditer(path.read_text(encoding="utf-8")):
            assert forbidden.search(match.group("body")) is None, (
                f"Executable documentation must be offline: {path}"
            )


def test_every_documentation_citation_has_a_central_bibliography_entry() -> None:
    citation = re.compile(r"\{cite:[^}]+\}`([^`]+)`")
    cited: set[str] = set()
    for path in (
        DOCS / "getting_started" / "index.md",
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
