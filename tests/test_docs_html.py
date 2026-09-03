from __future__ import annotations

import os
from pathlib import Path

import pytest
from tools.check_docs_html import (
    GUIDE_SLUGS,
    PRIMARY_NAVIGATION,
    _inventory_errors,
    _method_card_docnames,
    _navigation_errors,
    parse_page,
)

from believe14.registry import list_estimators

ROOT = Path(__file__).resolve().parents[1]

REPRESENTATIVE_SECTIONS = {
    "getting_started/index.html": "Getting started",
    "tutorials/index.html": "Tutorials",
    "guides/choosing-a-method.html": "Tutorials",
    "methods.html": "Methods",
    "examples/pca.html": "Methods",
    "api.html": "API reference",
    "development/index.html": "Development",
    "validation/index.html": "Development",
}


def _relative_href(page: Path, target: Path) -> str:
    return Path(os.path.relpath(target, page.parent)).as_posix()


def _navbar(
    site: Path,
    page: Path,
    labels: tuple[str, ...],
    active: str | None,
    *,
    more: bool,
) -> str:
    targets = dict(PRIMARY_NAVIGATION)
    entries: list[str] = []
    for label in labels:
        target = targets.get(label, "api.html")
        classes = "nav-item current active" if label == active else "nav-item"
        href = _relative_href(page, site / target)
        entries.append(
            f'<li class="{classes}"><a class="nav-link" href="{href}">{label}</a></li>'
        )
    if more:
        entries.append(
            '<li><button aria-controls="pst-nav-more-links">More</button>'
            '<ul id="pst-nav-more-links"></ul></li>'
        )
    return '<ul class="bd-navbar-elements navbar-nav">' + "".join(entries) + "</ul>"


def _write_site(
    site: Path,
    *,
    labels: tuple[str, ...] | None = None,
    wrong_active_page: str | None = None,
    more: bool = False,
    global_section_navigation: bool = False,
) -> dict[Path, object]:
    labels = labels or tuple(label for label, _ in PRIMARY_NAVIGATION)
    pages = {"index.html": None, **REPRESENTATIVE_SECTIONS}
    for relative_page, expected_active in pages.items():
        page = site / relative_page
        page.parent.mkdir(parents=True, exist_ok=True)
        active = expected_active
        if relative_page == wrong_active_page:
            active = "Tutorials"
        navbar = _navbar(site, page, labels, active, more=more)
        sidebar_class = "bd-sidebar-primary hide-on-wide"
        section_links = ""
        if relative_page not in {"index.html", "getting_started/index.html"}:
            sidebar_class = "bd-sidebar-primary"
            if global_section_navigation:
                targets = [site / "examples" / "pca.html"] + [
                    site / "guides" / f"{slug}.html" for slug in GUIDE_SLUGS
                ]
            else:
                targets = [page]
            section_links = (
                '<nav aria-label="Section Navigation">'
                + "".join(
                    f'<a href="{_relative_href(page, target)}">section</a>'
                    for target in targets
                )
                + "</nav>"
            )
        page.write_text(
            "<html><body>"
            + navbar
            + navbar
            + f'<div id="pst-primary-sidebar" class="{sidebar_class}">'
            + section_links
            + "</div></body></html>",
            encoding="utf-8",
        )
    return {page.resolve(): parse_page(page) for page in sorted(site.rglob("*.html"))}


def test_five_section_navigation_accepts_two_scoped_navbar_copies(
    tmp_path: Path,
) -> None:
    parsed = _write_site(tmp_path)

    assert _navigation_errors(tmp_path, parsed) == []
    assert all(len(page.primary_navigation) == 2 for page in parsed.values())


@pytest.mark.parametrize(
    "labels",
    [
        ("Getting started", "Tutorials", "Methods", "API reference"),
        (
            "Getting started",
            "Tutorials",
            "Methods",
            "API reference",
            "Development",
            "Extra",
        ),
        ("Tutorials", "Getting started", "Methods", "API reference", "Development"),
    ],
    ids=("missing", "extra", "reordered"),
)
def test_five_section_navigation_rejects_wrong_inventory(
    tmp_path: Path, labels: tuple[str, ...]
) -> None:
    parsed = _write_site(tmp_path, labels=labels)

    errors = _navigation_errors(tmp_path, parsed)

    assert any("has labels" in error for error in errors)


def test_five_section_navigation_rejects_wrong_active_section(tmp_path: Path) -> None:
    parsed = _write_site(tmp_path, wrong_active_page="examples/pca.html")

    errors = _navigation_errors(tmp_path, parsed)

    assert any(
        error.startswith("examples/pca.html:") and "active sections" in error
        for error in errors
    )


def test_five_section_navigation_rejects_more_overflow(tmp_path: Path) -> None:
    parsed = _write_site(tmp_path, more=True)

    assert any(
        "More overflow" in error for error in _navigation_errors(tmp_path, parsed)
    )


def test_section_navigation_rejects_global_method_and_guide_inventory(
    tmp_path: Path,
) -> None:
    parsed = _write_site(tmp_path, global_section_navigation=True)

    assert any(
        "instead of the current section" in error
        for error in _navigation_errors(tmp_path, parsed)
    )


def test_method_catalog_links_each_estimator_to_its_exact_card(tmp_path: Path) -> None:
    cards = _method_card_docnames(ROOT / "docs")
    methods = tmp_path / "methods.html"
    links: list[str] = []
    for info in list_estimators():
        target = tmp_path / f"{cards[info.name]}.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("<html></html>", encoding="utf-8")
        links.append(f'<a href="{target.relative_to(tmp_path)}">{info.name}</a>')
    methods.write_text(
        '<html><table class="believe14-catalog"><tr><td>'
        + "".join(links)
        + "</td></tr></table>"
        + "".join(links)
        + "</html>",
        encoding="utf-8",
    )
    parsed = {page.resolve(): parse_page(page) for page in tmp_path.rglob("*.html")}

    valid_errors = _inventory_errors(tmp_path, parsed)
    assert not [error for error in valid_errors if error.startswith("methods.html:")]

    wrong_links = links.copy()
    pca_index = next(
        index for index, info in enumerate(list_estimators()) if info.name == "PCA"
    )
    wrong_links[pca_index] = '<a href="api.html">PCA</a>'
    methods.write_text(
        '<html><table class="believe14-catalog"><tr><td>'
        + "".join(wrong_links)
        + "</td></tr></table>"
        + "".join(links)
        + "</html>",
        encoding="utf-8",
    )
    parsed[methods.resolve()] = parse_page(methods)

    invalid_errors = _inventory_errors(tmp_path, parsed)
    assert "methods.html: PCA does not link to examples/pca.html" in invalid_errors
