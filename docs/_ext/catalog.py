"""Registry-backed Sphinx directives for the believe14 documentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive

from believe14.api import Capability, EstimatorInfo
from believe14.registry import list_estimators

_FRONT_MATTER = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)
_FIELD = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_-]*):\s*(?P<value>.*?)\s*$")


@dataclass(frozen=True, slots=True)
class ExampleCard:
    """Metadata read from one executable method card."""

    estimator: str
    family: str
    path: Path

    @property
    def docname(self) -> str:
        return f"examples/{self.path.stem}"


def _front_matter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise ValueError(f"Method card has no YAML front matter: {path}")
    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        field = _FIELD.match(line)
        if field is not None:
            fields[field.group("key")] = field.group("value").strip("'\"")
    return fields


def discover_example_cards(docs_root: Path) -> tuple[ExampleCard, ...]:
    """Read and validate the complete registry-to-method-card mapping."""

    examples_root = docs_root / "examples"
    if not examples_root.is_dir():
        raise ValueError(f"Missing executable method-card directory: {examples_root}")

    cards: list[ExampleCard] = []
    for path in sorted(examples_root.glob("*.md")):
        fields = _front_matter(path)
        estimator = fields.get("believe14_estimator")
        family = fields.get("believe14_family")
        if estimator is None or family is None:
            raise ValueError(
                "Method-card front matter must define believe14_estimator and "
                f"believe14_family: {path}"
            )
        cards.append(ExampleCard(estimator=estimator, family=family, path=path))

    expected = {info.name: info for info in list_estimators()}
    by_name: dict[str, ExampleCard] = {}
    duplicates: list[str] = []
    for card in cards:
        if card.estimator in by_name:
            duplicates.append(card.estimator)
        by_name[card.estimator] = card

    missing = sorted(expected.keys() - by_name.keys())
    unexpected = sorted(by_name.keys() - expected.keys())
    mismatched = sorted(
        name
        for name in expected.keys() & by_name.keys()
        if by_name[name].family != expected[name].family
    )
    if duplicates or missing or unexpected or mismatched or len(cards) != 30:
        raise ValueError(
            "Executable method-card coverage does not match the public registry: "
            f"count={len(cards)}, duplicates={sorted(set(duplicates))}, "
            f"missing={missing}, unexpected={unexpected}, "
            f"family_mismatches={mismatched}"
        )
    return tuple(by_name[info.name] for info in list_estimators())


def _entry(value: str | nodes.Node) -> nodes.entry:
    entry = nodes.entry()
    if isinstance(value, str):
        entry += nodes.paragraph(text=value)
    else:
        paragraph = nodes.paragraph()
        paragraph += value
        entry += paragraph
    return entry


def _table(column_count: int, css_class: str) -> tuple[nodes.table, nodes.tbody]:
    table = nodes.table(classes=[css_class])
    group = nodes.tgroup(cols=column_count)
    table += group
    for _ in range(column_count):
        group += nodes.colspec(colwidth=1)
    body = nodes.tbody()
    group += body
    return table, body


def _add_header(table: nodes.table, labels: tuple[str, ...]) -> None:
    group = table[0]
    head = nodes.thead()
    group.insert(len(labels), head)
    row = nodes.row()
    head += row
    for label in labels:
        row += _entry(label)


def _mode(info: EstimatorInfo) -> str:
    if info.family == "estimation":
        return "dimension estimate"
    if Capability.TRANSFORM in info.capabilities:
        assert info.out_of_sample is not None
        return f"inductive ({info.out_of_sample})"
    return "transductive"


class CatalogDirective(Directive):
    """Render one believe14 family as a registry-backed table."""

    required_arguments = 1
    has_content = False

    def run(self):  # type: ignore[no-untyped-def]
        family = self.arguments[0]
        if family not in {"linear", "nonlinear", "estimation"}:
            raise self.error(f"Unknown believe14 family: {family}")
        table, body = _table(5, "believe14-catalog")
        _add_header(
            table,
            ("Estimator", "Approaches", "Supervision", "Out of sample", "Cost"),
        )
        for info in list_estimators(family=family):
            row = nodes.row()
            values = (
                info.name,
                ", ".join(sorted(info.approaches)),
                info.supervision,
                info.out_of_sample or "transductive / not applicable",
                info.complexity,
            )
            for value in values:
                row += _entry(value)
            body += row
        return [table]


class ExampleCoverageDirective(Directive):
    """Render the complete registry-to-executable-card coverage table."""

    has_content = False

    def run(self):  # type: ignore[no-untyped-def]
        environment = self.state.document.settings.env
        try:
            cards = discover_example_cards(Path(environment.srcdir))
        except ValueError as error:
            raise self.error(str(error)) from error

        builder = environment.app.builder
        table, body = _table(5, "believe14-example-coverage")
        _add_header(table, ("Estimator", "Family", "Supervision", "Mode", "Card"))
        for info, card in zip(list_estimators(), cards, strict=True):
            row = nodes.row(ids=[f"believe14-example-{info.name}"])
            target = builder.get_relative_uri(environment.docname, card.docname)
            link = nodes.reference("", "open example", refuri=target, internal=True)
            for value in (info.name, info.family, info.supervision, _mode(info), link):
                row += _entry(value)
            body += row
        return [table]


def _validate_examples(app) -> None:  # type: ignore[no-untyped-def]
    try:
        discover_example_cards(Path(app.srcdir))
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def setup(app):  # type: ignore[no-untyped-def]
    app.add_directive("believe14-catalog", CatalogDirective)
    app.add_directive("believe14-example-coverage", ExampleCoverageDirective)
    app.connect("builder-inited", _validate_examples)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
