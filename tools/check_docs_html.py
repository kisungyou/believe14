"""Audit rendered believe14 documentation, navigation, math, and public coverage."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from believe14.registry import list_estimators

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORED_TEXT_TAGS = {"code", "pre", "script", "style"}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
}
TEX_LEAK = re.compile(
    r"\\(?:begin|end|frac|lVert|mathbb|mathcal|operatorname|pi|rVert|sum|Theta|top)\b"
)
PROSE_IN_MATH = re.compile(
    r"[.`]\s+(?:A|An|For|It|Its|The|This|When|where|which|is|requires|to)\b"
)
TEXT_LIKE_COMMAND = re.compile(
    r"\\(?:text(?:normal|rm|sf|tt|bf|it)?|mbox|operatorname|"
    r"mathrm|mathbf|mathit|mathtt)\s*\{"
)
ENVIRONMENT_COMMAND = re.compile(r"\\(begin|end)\{([^{}]+)\}")
TEXT_ARGUMENT_SPECIALS = frozenset("_^&#%$")
GUIDE_SLUGS = (
    "choosing-a-method",
    "linear-latent-representations",
    "supervised-and-paired-reductions",
    "distance-and-graph-embeddings",
    "stress-and-stochastic-embeddings",
    "intrinsic-dimension",
)
PRIMARY_NAVIGATION = (
    ("Getting started", "getting_started/index.html"),
    ("Tutorials", "tutorials/index.html"),
    ("Methods", "methods.html"),
    ("API reference", "api.html"),
    ("Development", "development/index.html"),
)
CARD_FRONT_MATTER = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)
CARD_FIELD = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_-]*):\s*(?P<value>.*?)\s*$")


@dataclass(frozen=True, slots=True)
class NavigationEntry:
    """One rendered navigation link."""

    label: str
    href: str
    active: bool


class RenderedPageParser(HTMLParser):
    """Collect rendered prose, math nodes, references, and element IDs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool, bool, frozenset[str]]] = []
        self.ignored_depth = 0
        self.math_depth = 0
        self.current_math: list[str] = []
        self.math_nodes: list[str] = []
        self.prose: list[str] = []
        self.references: list[tuple[str, str]] = []
        self.catalog_links: list[NavigationEntry] = []
        self._catalog_depth = 0
        self._catalog_link: tuple[str, bool] | None = None
        self._catalog_text: list[str] = []
        self.ids: set[str] = set()
        self.primary_navigation: list[list[NavigationEntry]] = []
        self._navigation_depth = 0
        self._navigation_link: tuple[str, bool] | None = None
        self._navigation_text: list[str] = []
        self.primary_sidebar_classes: set[str] | None = None
        self.section_navigation: list[NavigationEntry] = []
        self.has_more_overflow = False
        self._section_navigation_depth = 0
        self._section_link: tuple[str, bool] | None = None
        self._section_text: list[str] = []

    def _attributes(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        element_id = attrs.get("id")
        if element_id:
            self.ids.add(element_id)
        if tag in {"a", "link"} and attrs.get("href"):
            self.references.append(("href", attrs["href"]))
        if tag in {"img", "script", "source"} and attrs.get("src"):
            self.references.append(("src", attrs["src"]))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._attributes(tag, attrs)
        if tag in VOID_TAGS:
            return
        attributes = dict(attrs)
        classes = frozenset((attributes.get("class") or "").split())
        if (
            attributes.get("id") == "pst-nav-more-links"
            or attributes.get("aria-controls") == "pst-nav-more-links"
        ):
            self.has_more_overflow = True
        starts_ignored = tag in IGNORED_TEXT_TAGS
        starts_math = "math" in classes and "nohighlight" in classes
        if "believe14-catalog" in classes:
            self._catalog_depth = 1
        elif self._catalog_depth:
            self._catalog_depth += 1
        if self._catalog_depth and tag == "a":
            active = any(
                open_tag == "li" and bool({"active", "current"} & open_classes)
                for open_tag, _, _, open_classes in self.stack
            )
            self._catalog_link = (attributes.get("href") or "", active)
            self._catalog_text = []
        if tag == "ul" and {"bd-navbar-elements", "navbar-nav"} <= classes:
            self.primary_navigation.append([])
            self._navigation_depth = 1
        elif self._navigation_depth:
            self._navigation_depth += 1
        if self._navigation_depth and tag == "a":
            active = any(
                open_tag == "li" and bool({"active", "current"} & open_classes)
                for open_tag, _, _, open_classes in self.stack
            )
            self._navigation_link = (attributes.get("href") or "", active)
            self._navigation_text = []

        if tag == "div" and attributes.get("id") == "pst-primary-sidebar":
            self.primary_sidebar_classes = set(classes)
        if tag == "nav" and attributes.get("aria-label") == "Section Navigation":
            self._section_navigation_depth = 1
        elif self._section_navigation_depth:
            self._section_navigation_depth += 1
        if self._section_navigation_depth and tag == "a":
            active = any(
                open_tag == "li" and bool({"active", "current"} & open_classes)
                for open_tag, _, _, open_classes in self.stack
            )
            self._section_link = (attributes.get("href") or "", active)
            self._section_text = []

        self.stack.append((tag, starts_ignored, starts_math, classes))
        if starts_ignored:
            self.ignored_depth += 1
        if starts_math:
            if self.math_depth == 0:
                self.current_math = []
            self.math_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._attributes(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            return
        open_tag, ends_ignored, ends_math, _ = self.stack.pop()
        if open_tag != tag:
            return
        if tag == "a" and self._catalog_link is not None:
            href, active = self._catalog_link
            label = " ".join("".join(self._catalog_text).split())
            self.catalog_links.append(NavigationEntry(label, href, active))
            self._catalog_link = None
            self._catalog_text = []
        if self._catalog_depth:
            self._catalog_depth -= 1
        if tag == "a" and self._navigation_link is not None:
            href, active = self._navigation_link
            label = " ".join("".join(self._navigation_text).split())
            self.primary_navigation[-1].append(NavigationEntry(label, href, active))
            self._navigation_link = None
            self._navigation_text = []
        if self._navigation_depth:
            self._navigation_depth -= 1

        if tag == "a" and self._section_link is not None:
            href, active = self._section_link
            label = " ".join("".join(self._section_text).split())
            self.section_navigation.append(NavigationEntry(label, href, active))
            self._section_link = None
            self._section_text = []
        if self._section_navigation_depth:
            self._section_navigation_depth -= 1
        if ends_ignored:
            self.ignored_depth -= 1
        if ends_math:
            self.math_depth -= 1
            if self.math_depth == 0:
                self.math_nodes.append("".join(self.current_math).strip())
                self.current_math = []

    def handle_data(self, data: str) -> None:
        if self._catalog_link is not None:
            self._catalog_text.append(data)
        if self._navigation_link is not None:
            self._navigation_text.append(data)
        if self._section_link is not None:
            self._section_text.append(data)
        if self.ignored_depth:
            return
        if self.math_depth:
            self.current_math.append(data)
        else:
            self.prose.append(data)


def parse_page(path: Path) -> RenderedPageParser:
    """Parse one rendered HTML page."""

    parser = RenderedPageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def _method_card_docnames(docs_root: Path) -> dict[str, str]:
    """Return the validated estimator-to-card mapping without Sphinx imports."""

    observed: dict[str, tuple[str, str]] = {}
    for path in sorted((docs_root / "examples").glob("*.md")):
        match = CARD_FRONT_MATTER.match(path.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"method card has no front matter: {path}")
        fields: dict[str, str] = {}
        for line in match.group("body").splitlines():
            field = CARD_FIELD.match(line)
            if field is not None:
                fields[field.group("key")] = field.group("value").strip("'\"")
        estimator = fields.get("believe14_estimator")
        family = fields.get("believe14_family")
        if estimator is None or family is None:
            raise ValueError(f"method card metadata is incomplete: {path}")
        if estimator in observed:
            raise ValueError(f"duplicate method card for {estimator}")
        observed[estimator] = (family, f"examples/{path.stem}")

    expected = {info.name: info.family for info in list_estimators()}
    observed_families = {name: family for name, (family, _) in observed.items()}
    if len(observed) != 30 or observed_families != expected:
        raise ValueError("method cards do not exactly match the public registry")
    return {name: docname for name, (_, docname) in observed.items()}


def _is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        slash_count += 1
        index -= 1
    return slash_count % 2 == 1


def _braced_argument(text: str, opening_brace: int) -> tuple[str | None, int]:
    depth = 1
    index = opening_brace + 1
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[opening_brace + 1 : index], index
        index += 1
    return None, len(text)


def tex_syntax_errors(tex: str) -> list[str]:
    """Return structural TeX errors detectable without a browser runtime."""

    errors: list[str] = []
    brace_depth = 0
    for index, character in enumerate(tex):
        if _is_escaped(tex, index):
            continue
        if character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
            if brace_depth < 0:
                errors.append("unmatched closing brace")
                brace_depth = 0
    if brace_depth:
        errors.append(f"{brace_depth} unmatched opening brace(s)")

    environments: list[str] = []
    for match in ENVIRONMENT_COMMAND.finditer(tex):
        action, environment = match.groups()
        if action == "begin":
            environments.append(environment)
        elif not environments:
            errors.append(f"unexpected end of {environment!r} environment")
        elif environments[-1] != environment:
            errors.append(f"environment {environments[-1]!r} closed by {environment!r}")
            environments.pop()
        else:
            environments.pop()
    errors.extend(
        f"unclosed {environment!r} environment" for environment in environments
    )

    for match in TEXT_LIKE_COMMAND.finditer(tex):
        argument, _ = _braced_argument(tex, match.end() - 1)
        if argument is None:
            errors.append(f"unclosed argument for {match.group(0)[:-1]!r}")
            continue
        unsafe = sorted(
            {
                character
                for index, character in enumerate(argument)
                if character in TEXT_ARGUMENT_SPECIALS
                and not _is_escaped(argument, index)
            }
        )
        if unsafe:
            rendered = ", ".join(repr(character) for character in unsafe)
            errors.append(
                f"{match.group(0)[:-1]} argument contains unescaped TeX "
                f"special(s): {rendered}"
            )
    if re.search(r"(?<!\\)\$", tex):
        errors.append("unescaped dollar sign inside a math node")
    return errors


def resolve_local_reference(
    site: Path, page: Path, reference: str
) -> tuple[Path, str] | None:
    """Resolve a local rendered-page reference, preserving its fragment."""

    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or reference.startswith("//"):
        return None
    path_text = unquote(parsed.path)
    if not path_text:
        target = page
    elif path_text.startswith("/"):
        target = site / path_text.lstrip("/")
    else:
        target = page.parent / path_text
    if path_text.endswith("/"):
        target = target / "index.html"
    return target.resolve(), unquote(parsed.fragment)


def _relative_page(site: Path, page: Path) -> str:
    return page.relative_to(site).as_posix()


def _navigation_errors(
    site: Path, parsed_pages: dict[Path, RenderedPageParser]
) -> list[str]:
    """Validate the five-section site navigation and section-level scoping."""

    errors: list[str] = []
    expected_labels = tuple(label for label, _ in PRIMARY_NAVIGATION)
    expected_targets = tuple(
        (site / target).resolve() for _, target in PRIMARY_NAVIGATION
    )

    for page, parser in parsed_pages.items():
        relative_page = _relative_page(site, page)
        if parser.has_more_overflow:
            errors.append(
                f"{relative_page}: primary navbar contains a More overflow menu"
            )
        if len(parser.primary_navigation) != 2:
            errors.append(
                f"{relative_page}: expected two primary navbar copies, found "
                f"{len(parser.primary_navigation)}"
            )
            continue
        for copy_number, navigation in enumerate(parser.primary_navigation, 1):
            labels = tuple(entry.label for entry in navigation)
            if labels != expected_labels:
                errors.append(
                    f"{relative_page}: primary navbar copy {copy_number} has "
                    f"labels {labels!r}, expected {expected_labels!r}"
                )
                continue
            targets = tuple(
                resolved[0]
                if (resolved := resolve_local_reference(site, page, entry.href))
                else None
                for entry in navigation
            )
            if targets != expected_targets:
                errors.append(
                    f"{relative_page}: primary navbar copy {copy_number} has "
                    "incorrect section targets"
                )

    representative_sections = {
        "getting_started/index.html": "Getting started",
        "tutorials/index.html": "Tutorials",
        "guides/choosing-a-method.html": "Tutorials",
        "methods.html": "Methods",
        "examples/pca.html": "Methods",
        "api.html": "API reference",
        "development/index.html": "Development",
        "validation/index.html": "Development",
    }
    for relative_page, expected_active in representative_sections.items():
        page = (site / relative_page).resolve()
        parser = parsed_pages.get(page)
        if parser is None:
            errors.append(f"Missing representative navigation page: {relative_page}")
            continue
        for copy_number, navigation in enumerate(parser.primary_navigation, 1):
            active = tuple(entry.label for entry in navigation if entry.active)
            if active != (expected_active,):
                errors.append(
                    f"{relative_page}: primary navbar copy {copy_number} has active "
                    f"sections {active!r}, expected {(expected_active,)!r}"
                )

    for relative_page in ("index.html", "getting_started/index.html"):
        page = (site / relative_page).resolve()
        parser = parsed_pages.get(page)
        if parser is None:
            continue
        sidebar_classes = parser.primary_sidebar_classes
        if sidebar_classes is not None and "hide-on-wide" not in sidebar_classes:
            errors.append(f"{relative_page}: desktop primary sidebar must be hidden")
        if parser.section_navigation:
            errors.append(f"{relative_page}: section navigation must be empty")

    card_targets = {page.resolve() for page in (site / "examples").glob("*.html")}
    guide_targets = {
        (site / "guides" / f"{slug}.html").resolve() for slug in GUIDE_SLUGS
    }
    for page, parser in parsed_pages.items():
        section_targets = {
            resolved[0]
            for entry in parser.section_navigation
            if (resolved := resolve_local_reference(site, page, entry.href)) is not None
        }
        if card_targets <= section_targets and guide_targets <= section_targets:
            errors.append(
                f"{_relative_page(site, page)}: section navigation exposes every "
                "method card and guide instead of the current section"
            )
    return errors


def _inventory_errors(
    site: Path, parsed_pages: dict[Path, RenderedPageParser]
) -> list[str]:
    errors: list[str] = []
    api_page = (site / "api.html").resolve()
    if api_page not in parsed_pages:
        errors.append(f"Missing generated API page: {api_page}")
    else:
        expected_api = {
            f"believe14.{info.family}.{info.name}" for info in list_estimators()
        }
        missing_api = sorted(expected_api - parsed_pages[api_page].ids)
        if missing_api:
            errors.append(f"api.html: missing public anchors {missing_api}")
        rendered = api_page.read_text(encoding="utf-8")
        if ".. py:class::" in rendered or ".. py:module::" in rendered:
            errors.append("api.html: autodoc directives were emitted as plain text")

    methods_page = (site / "methods.html").resolve()
    if methods_page not in parsed_pages:
        errors.append(f"Missing generated method catalog: {methods_page}")
    else:
        source_root = PROJECT_ROOT / "docs"
        cards = _method_card_docnames(source_root)
        links = parsed_pages[methods_page].catalog_links
        for info in list_estimators():
            matching = [link for link in links if link.label == info.name]
            expected_target = (site / f"{cards[info.name]}.html").resolve()
            if len(matching) != 1:
                errors.append(
                    f"methods.html: expected one catalog link for {info.name}, "
                    f"found {len(matching)}"
                )
                continue
            resolved = resolve_local_reference(site, methods_page, matching[0].href)
            if resolved is None or resolved[0] != expected_target:
                expected_relative = expected_target.relative_to(site).as_posix()
                errors.append(
                    f"methods.html: {info.name} does not link to {expected_relative}"
                )
            elif not expected_target.exists():
                expected_relative = expected_target.relative_to(site).as_posix()
                errors.append(
                    f"methods.html: linked card for {info.name} does not exist: "
                    f"{expected_relative}"
                )

    for slug in GUIDE_SLUGS:
        guide = (site / "guides" / f"{slug}.html").resolve()
        if guide not in parsed_pages:
            errors.append(f"Missing executable guide page: guides/{slug}.html")

    example_pages = sorted((site / "examples").glob("*.html"))
    if len(example_pages) != 30:
        errors.append(
            f"Expected exactly 30 rendered method cards, found {len(example_pages)}"
        )
    for page in example_pages:
        rendered = page.read_text(encoding="utf-8")
        if "cell_output" not in rendered:
            errors.append(f"{page.relative_to(site)}: no executed notebook output")

    choosing = (site / "guides" / "choosing-a-method.html").resolve()
    if choosing in parsed_pages:
        expected_rows = {f"believe14-example-{info.name}" for info in list_estimators()}
        missing_rows = sorted(expected_rows - parsed_pages[choosing].ids)
        if missing_rows:
            errors.append(
                "guides/choosing-a-method.html: missing registry coverage rows "
                f"{missing_rows}"
            )
    return errors


def audit_site(site: Path) -> list[str]:
    """Return all rendered-site errors without stopping at the first failure."""

    pages = sorted(path for path in site.rglob("*.html") if "_static" not in path.parts)
    if not pages:
        return [f"No HTML pages found under {site}."]
    parsed_pages = {page.resolve(): parse_page(page) for page in pages}
    errors = _inventory_errors(site, parsed_pages)
    errors.extend(_navigation_errors(site, parsed_pages))
    math_count = 0
    local_reference_count = 0

    for page, parser in parsed_pages.items():
        relative_page = page.relative_to(site)
        raw_html = page.read_text(encoding="utf-8")
        if re.search(r'class="[^"]*\btraceback\b', raw_html):
            errors.append(f"{relative_page}: executed notebook traceback is embedded")
        prose = " ".join(parser.prose)
        if TEX_LEAK.search(prose):
            errors.append(f"{relative_page}: raw TeX command leaked into prose")
        if re.search(r"(?<!\\)\$", prose):
            errors.append(f"{relative_page}: dollar delimiter leaked into prose")

        for math in parser.math_nodes:
            math_count += 1
            is_inline = math.startswith(r"\(")
            expected_end = r"\)" if is_inline else r"\]"
            if not math.endswith(expected_end):
                errors.append(
                    f"{relative_page}: unterminated rendered math node {math[:100]!r}"
                )
                continue
            if is_inline and len(math) > 180:
                errors.append(
                    f"{relative_page}: suspiciously long inline math node "
                    f"{math[:100]!r}"
                )
            if "`" in math or PROSE_IN_MATH.search(math):
                errors.append(
                    f"{relative_page}: prose appears inside math node {math[:100]!r}"
                )
            errors.extend(
                f"{relative_page}: {issue} in rendered math node {math[:100]!r}"
                for issue in tex_syntax_errors(math[2:-2])
            )

        for kind, reference in parser.references:
            resolved = resolve_local_reference(site, page, reference)
            if resolved is None:
                continue
            local_reference_count += 1
            target, fragment = resolved
            if not target.exists():
                errors.append(f"{relative_page}: broken {kind} target {reference!r}")
                continue
            if fragment and target.suffix == ".html":
                target_parser = parsed_pages.get(target)
                if target_parser is None:
                    target_parser = parse_page(target)
                    parsed_pages[target] = target_parser
                if fragment not in target_parser.ids:
                    errors.append(f"{relative_page}: missing fragment {reference!r}")

    if math_count == 0:
        errors.append(
            "No rendered math nodes were found; the MathJax audit was not exercised"
        )
    if not errors:
        print(
            f"Audited {len(pages)} HTML pages, {math_count} math nodes, "
            f"{local_reference_count} local references, 30 API anchors, "
            "30 executed method cards, and 6 guides."
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    arguments = parser.parse_args()
    site = arguments.build_dir.resolve()
    errors = audit_site(site)
    if errors:
        raise SystemExit("Rendered documentation audit failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
