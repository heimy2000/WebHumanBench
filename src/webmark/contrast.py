"""Static CSS contrast guardrail utilities.

This module implements a conservative, dependency-light proxy for contrast
regressions. It is not a replacement for browser + axe audits, because it does
not evaluate cascade, pseudo-states, media queries, or rendered geometry. Its
role is narrower: prevent the search from accepting CSS candidates that
obviously increase low-contrast foreground/background pairs in inline styles or
simple style blocks.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

RGB = tuple[int, int, int]

_RE_STYLE_BLOCK = re.compile(r"<style[^>]*>(.*?)</style>", re.IGNORECASE | re.DOTALL)
_RE_RULE = re.compile(r"([^{}]+)\{([^{}]+)\}", re.DOTALL)
_RE_INLINE_STYLE = re.compile(r"style\s*=\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL)
_RE_DECL = re.compile(r"([A-Za-z-]+)\s*:\s*([^;{}]+)")
_RE_HEX = re.compile(r"#[0-9a-fA-F]{3,8}")
_RE_RGB = re.compile(r"rgba?\(([^)]+)\)", re.IGNORECASE)
_TEXTISH_SELECTORS = (
    "body", "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "button", "a", "li", "td", "th", "label", "span", "section",
    ".hero", ".btn", ".cta", ".card", ".item", ".alert",
)


def static_low_contrast_exposure(html: str) -> int:
    """Estimate the number of simple CSS text/background pairs below WCAG AA.

    The count is intentionally approximate. A candidate with a higher count is
    treated as riskier, but a count of zero is not an accessibility pass.
    """

    body_defaults = _body_defaults(html)
    exposures = 0
    for selector, declarations in _style_rules(html):
        if not _selector_may_contain_text(selector):
            continue
        exposures += _declaration_exposure(declarations, body_defaults, selector)
    for declarations in _inline_styles(html):
        exposures += _declaration_exposure(declarations, body_defaults, "")
    return exposures


def contrast_exposure_delta(initial_html: str, corrected_html: str) -> int:
    """Return corrected minus initial static low-contrast exposure."""

    return static_low_contrast_exposure(corrected_html) - static_low_contrast_exposure(initial_html)


def _style_rules(html: str) -> list[tuple[str, dict[str, str]]]:
    rules: list[tuple[str, dict[str, str]]] = []
    for block in _RE_STYLE_BLOCK.findall(html):
        for selector, body in _RE_RULE.findall(block):
            rules.append((selector.strip().lower(), _parse_declarations(body)))
    return rules


def _inline_styles(html: str) -> list[dict[str, str]]:
    return [_parse_declarations(style) for _quote, style in _RE_INLINE_STYLE.findall(html)]


def _parse_declarations(css: str) -> dict[str, str]:
    return {name.strip().lower(): value.strip() for name, value in _RE_DECL.findall(css)}


def _body_defaults(html: str) -> dict[str, Sequence[RGB]]:
    fg: Sequence[RGB] = ((0, 0, 0),)
    bg: Sequence[RGB] = ((255, 255, 255),)
    for selector, declarations in _style_rules(html):
        selectors = [part.strip() for part in selector.split(",")]
        if "body" not in selectors:
            continue
        fg = _colors_from_value(declarations.get("color")) or fg
        bg = (
            _colors_from_value(declarations.get("background-color"))
            or _colors_from_value(declarations.get("background"))
            or bg
        )
        break
    return {"color": fg, "background": bg}


def _declaration_exposure(
    declarations: dict[str, str],
    defaults: dict[str, Sequence[RGB]],
    selector: str,
) -> int:
    fg = _colors_from_value(declarations.get("color")) or defaults["color"]
    bg = (
        _colors_from_value(declarations.get("background-color"))
        or _colors_from_value(declarations.get("background"))
        or defaults["background"]
    )
    if not fg or not bg:
        return 0
    min_ratio = min(_contrast_ratio(a, b) for a in fg for b in bg)
    threshold = 3.0 if _large_text_like(selector, declarations) else 4.5
    return int(min_ratio < threshold)


def _selector_may_contain_text(selector: str) -> bool:
    selectors = [part.strip() for part in selector.split(",")]
    return any(any(token in part for token in _TEXTISH_SELECTORS) for part in selectors)


def _large_text_like(selector: str, declarations: dict[str, str]) -> bool:
    if any(tag in selector for tag in ("h1", "h2", ".hero")):
        return True
    size = declarations.get("font-size")
    if not size:
        return False
    match = re.search(r"([\d.]+)\s*(px|rem|em)?", size)
    if not match:
        return False
    value = float(match.group(1))
    unit = match.group(2) or "px"
    if unit == "px":
        return value >= 18.0
    return value >= 1.125


def _colors_from_value(value: str | None) -> list[RGB]:
    if not value:
        return []
    colors: list[RGB] = []
    for raw in _RE_HEX.findall(value):
        parsed = _parse_hex(raw)
        if parsed is not None:
            colors.append(parsed)
    for raw in _RE_RGB.findall(value):
        parsed = _parse_rgb(raw)
        if parsed is not None:
            colors.append(parsed)
    return colors


def _parse_hex(raw: str) -> RGB | None:
    value = raw.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) not in (6, 8):
        return None
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:
        return None


def _parse_rgb(raw: str) -> RGB | None:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) < 3:
        return None
    try:
        rgb = tuple(max(0, min(255, int(float(part)))) for part in parts[:3])
    except ValueError:
        return None
    return rgb  # type: ignore[return-value]


def _relative_luminance(rgb: RGB) -> float:
    values = []
    for channel in rgb:
        c = channel / 255.0
        values.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]


def _contrast_ratio(a: RGB, b: RGB) -> float:
    la = _relative_luminance(a)
    lb = _relative_luminance(b)
    high, low = max(la, lb), min(la, lb)
    return (high + 0.05) / (low + 0.05)
