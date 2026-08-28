"""Audit rendered believe14 documentation, navigation, math, and public coverage."""

from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from believe14.registry import list_estimators

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


class RenderedPageParser(HTMLParser):
    """Collect rendered prose, math nodes, references, and element IDs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool, bool]] = []
        self.ignored_depth = 0
        self.math_depth = 0
        self.current_math: list[str] = []
        self.math_nodes: list[str] = []
        self.prose: list[str] = []
        self.references: list[tuple[str, str]] = []
        self.ids: set[str] = set()

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
        classes = set((dict(attrs).get("class") or "").split())
        starts_ignored = tag in IGNORED_TEXT_TAGS
        starts_math = "math" in classes and "nohighlight" in classes
        self.stack.append((tag, starts_ignored, starts_math))
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
        open_tag, ends_ignored, ends_math = self.stack.pop()
        if open_tag != tag:
            return
        if ends_ignored:
            self.ignored_depth -= 1
        if ends_math:
            self.math_depth -= 1
            if self.math_depth == 0:
                self.math_nodes.append("".join(self.current_math).strip())
                self.current_math = []

    def handle_data(self, data: str) -> None:
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
