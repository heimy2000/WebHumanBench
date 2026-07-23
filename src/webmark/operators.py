"""Eight atomic CSS-level correction operators.

Each operator is a pure HTML-to-HTML transform. The three scored release
operators are ``font_scale``, ``spacing``, and ``color``. ``radius`` remains
available as a diagnostic transform, but it is not part of the scored release
search because the released reference does not contain a radius dimension.

All operators compose via :func:`apply_operator` and never mutate inputs in
place; the canonical ``apply_operator_chain`` helper applies a sequence in
documented order.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

# ── canonical operator names ─────────────────────────────────────────────────
PRIMARY_OPERATORS: tuple = ("font_scale", "spacing", "color")
ABLATION_OPERATORS: tuple = (
    "radius",
    "hero",
    "body_readability",
    "accent",
    "typeface_swap",
)
OPERATOR_CATALOG: tuple = PRIMARY_OPERATORS + ABLATION_OPERATORS
OperatorName = str  # alias used in type hints


# ── regex helpers (compile-once) ────────────────────────────────────────────
_RE_FONT_SIZE = re.compile(r"font-size\s*:\s*([\d.]+)(px|em|rem)", re.IGNORECASE)
_RE_LINE_HEIGHT = re.compile(r"line-height\s*:\s*([\d.]+)", re.IGNORECASE)
_RE_LINE_HEIGHT_UNITS = re.compile(
    r"line-height\s*:\s*([\d.]+)(px|em|rem)?", re.IGNORECASE
)
_RE_BORDER_RADIUS = re.compile(r"border-radius\s*:\s*([\d.]+)(px|em|rem|%)?", re.IGNORECASE)
_RE_H1 = re.compile(r"(<h1[^>]*style\s*=\s*\")[^\"]*(\">)", re.IGNORECASE)
_RE_BODY_STYLE = re.compile(r"(<body[^>]*style\s*=\s*\")[^\"]*(\">)", re.IGNORECASE)
_RE_RAW_HTML_REGION = re.compile(
    r"(?P<style>"
    r"(?P<style_open><style\b[^>]*>)"
    r"(?P<style_css>.*?)"
    r"(?P<style_close></style\s*>)"
    r")"
    r"|(?P<script><script\b[^>]*>.*?</script\s*>)"
    r"|(?P<comment><!--.*?-->)",
    re.IGNORECASE | re.DOTALL,
)
_RE_HTML_TAG = re.compile(r"<(?:\"[^\"]*\"|'[^']*'|[^'\">])*>", re.DOTALL)
_RE_STYLE_ATTRIBUTE = re.compile(
    r"(?<![-:\w])(?P<prefix>style\s*=\s*)(?P<quote>[\"'])(?P<css>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)


def _transform_css_contexts(html: str, transform: Callable[[str], str]) -> str:
    """Apply a declaration transform only inside style elements/attributes."""

    def transform_attributes(fragment: str) -> str:
        def transform_tag(tag_match: re.Match) -> str:
            def transform_attribute(attribute_match: re.Match) -> str:
                quote = attribute_match.group("quote")
                return (
                    f"{attribute_match.group('prefix')}{quote}"
                    f"{transform(attribute_match.group('css'))}{quote}"
                )

            return _RE_STYLE_ATTRIBUTE.sub(transform_attribute, tag_match.group(0))

        return _RE_HTML_TAG.sub(transform_tag, fragment)

    chunks: list[str] = []
    cursor = 0
    for match in _RE_RAW_HTML_REGION.finditer(html):
        chunks.append(transform_attributes(html[cursor : match.start()]))
        if match.group("style") is not None:
            chunks.extend(
                (
                    match.group("style_open"),
                    transform(match.group("style_css")),
                    match.group("style_close"),
                )
            )
        else:
            chunks.append(match.group(0))
        cursor = match.end()
    chunks.append(transform_attributes(html[cursor:]))
    return "".join(chunks)


def _multiply_property(
    html: str,
    pattern: re.Pattern[str],
    units: Sequence[str] = ("px", "em", "rem"),
    factor: float = 1.0,
) -> str:
    """Generic helper: match ``<prop>:<n><unit>`` and replace with ``<n*factor><unit>``."""

    def repl(m: re.Match) -> str:
        full = m.group(0)
        raw_value = m.group(1)
        unit = m.group(2)
        if unit not in units:
            return full
        old = float(raw_value)
        new = round(old * factor, 3)
        # Preserve original formatting (integer vs decimal); if the
        # original was an integer literal, render as int when possible.
        if "." not in raw_value and new == int(new):
            new_str = str(int(new))
        else:
            new_str = f"{new:g}"
        return full.replace(f"{raw_value}{unit}", f"{new_str}{unit}", 1)

    return pattern.sub(repl, html)


# ── the eight operators ──────────────────────────────────────────────────────
def font_scale(html: str, *, factor: float = 1.18) -> str:
    """``font_scale`` operator. Multiplies all ``font-size`` declarations by ``factor``.

    The factor is applied globally so per-element font-size ratios remain
    proportional when all declarations use supported units.
    """
    return _transform_css_contexts(
        html, lambda css: _multiply_property(css, _RE_FONT_SIZE, factor=factor)
    )


def spacing(html: str, *, factor: float = 1.30) -> str:
    """``spacing`` operator. Multiplies all ``line-height`` by ``factor``.

    Unitless values and optional ``px``, ``em``, or ``rem`` units are
    supported. Existing units are preserved.
    """

    def repl(m: re.Match) -> str:
        full = m.group(0)
        raw_value = m.group(1)
        unit = m.group(2) or ""
        try:
            old = float(raw_value)
        except ValueError:
            return full
        new = round(old * factor, 3)
        if "." not in raw_value and new == int(new):
            new_str = str(int(new))
        else:
            new_str = f"{new:g}"
        # Drop the unit if none was present in the original; CSS
        # interprets unitless line-height as a multiplier (preferred).
        if unit:
            return full.replace(f"{raw_value}{unit}", f"{new_str}{unit}", 1)
        return full.replace(raw_value, new_str, 1)

    return _transform_css_contexts(html, lambda css: _RE_LINE_HEIGHT_UNITS.sub(repl, css))


def radius(html: str, *, factor: float = 1.20) -> str:
    """``radius`` operator. Multiplies ``border-radius`` by ``factor``.

    This transform remains diagnostic because radius is not represented in
    the released scoring objective.
    """
    return _transform_css_contexts(
        html, lambda css: _multiply_property(css, _RE_BORDER_RADIUS, factor=factor)
    )


def color(html: str, *, seed: int = 42) -> str:
    """``color`` operator. Re-seed the palette while preserving luminance roles.

    The transform is deterministic given ``seed``. The simple strategy
    used here maps text, background, border, and custom-property colours
    separately. Dark colours remain dark, light colours remain light, and
    accent colours are mapped to a fixed accent set. This is a deterministic
    role-aware mapping, not a contrast or accessibility guarantee.
    """
    return _transform_css_contexts(
        html, lambda css: _RE_COLOR_DECL.sub(lambda m: _map_color_declaration(m, seed), css)
    )


# ── ablation operators (SE-5) ─────────────────────────────────────────────────
def hero(html: str, *, size: int = 72) -> str:
    """Inject a hero section above the body. ``size`` controls the headline px."""

    body_match = re.search(r"<body[^>]*>", html, re.IGNORECASE)
    if not body_match:
        return html
    pos = body_match.end()
    hero_html = (
        f'<section class="hero" style="padding:80px 0;background:#1a1a2e;'
        f'color:#fff;text-align:center">'
        f'<h1 style="font-size:{size}px;font-weight:700;margin:0;line-height:1.1">'
        f'Hero Title</h1></section>'
    )
    return html[:pos] + hero_html + html[pos:]


def body_readability(html: str, *, body_size: int = 16) -> str:
    """Inject a ``body{font-size:Xpx}`` rule via the first ``<style>`` block."""

    rule = f"body {{ font-size: {body_size}px; }}"
    if rule in html:
        return html
    style_match = re.search(r"(<style[^>]*>)(.*?)(</style>)", html, re.DOTALL | re.IGNORECASE)
    if style_match:
        return html.replace(style_match.group(3), rule + style_match.group(3), 1)
    return f"<style>{rule}</style>\n{html}"


def accent(html: str, *, accent: str = "#2C7BE5") -> str:
    """Append a primary-accent colour to the palette (one entry)."""

    if accent in html:
        return html
    style_match = re.search(r"(</style>)", html, re.IGNORECASE)
    if style_match:
        return html.replace(
            style_match.group(1),
            f":root {{ --accent: {accent}; }}\n" + style_match.group(1),
            1,
        )
    return f"<style>:root {{ --accent: {accent}; }}</style>\n{html}"


def typeface_swap(html: str, *, family: str = "Inter, system-ui, sans-serif") -> str:
    """Replace ``font-family`` with one caller-selected family stack.

    Handles multi-word quoted families like ``'Inter', serif`` by
    matching up to the next ``;`` or end-of-style-block delimiter.
    """

    # Use a non-greedy match until a declaration terminator (`;`).
    pattern = re.compile(r"font-family\s*:[^;]+?(?=;|\n|$)", re.IGNORECASE)
    return _transform_css_contexts(
        html, lambda css: pattern.sub(f"font-family: {family}", css)
    )


# ── dispatch ─────────────────────────────────────────────────────────────────
_OPERATOR_FUNCTIONS: dict[OperatorName, Callable[..., str]] = {
    "font_scale": font_scale,
    "spacing": spacing,
    "radius": radius,
    "color": color,
    "hero": hero,
    "body_readability": body_readability,
    "accent": accent,
    "typeface_swap": typeface_swap,
}


def apply_operator(
    html: str,
    name: OperatorName,
    **kwargs,
) -> str:
    """Apply a single operator. ``kwargs`` are forwarded to the operator."""

    fn = _OPERATOR_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(
            f"unknown operator {name!r}; choose from {list(_OPERATOR_FUNCTIONS)}"
        )
    return fn(html, **kwargs)


def apply_operator_chain(
    html: str,
    chain: Sequence[OperatorName],
    **shared_kwargs,
) -> str:
    """Apply a sequence of operators, in documented left-to-right order."""

    out = html
    for name in chain:
        out = apply_operator(out, name, **shared_kwargs.get(name, {}))
    return out


# ── palette generator (shared by ``color``) ─────────────────────────────────
_RE_COLOR_DECL = re.compile(
    r"([A-Za-z-]+)\s*:\s*([^;{}]*(?:#[0-9a-fA-F]{3,8}|rgba?\([^)]+\))[^;{}]*)",
    re.IGNORECASE,
)
_RE_COLOR_TOKEN = re.compile(r"#[0-9a-fA-F]{3,8}|rgba?\([^)]+\)", re.IGNORECASE)


def _palette(seed: int) -> list[str]:
    """Deterministic 12-entry low-saturation palette."""

    base = [
        "#0F172A", "#1E293B", "#334155", "#475569",
        "#1E40AF", "#1D4ED8", "#2563EB", "#3B82F6",
        "#047857", "#059669", "#10B981", "#34D399",
    ]
    if seed != 42:
        # Deterministic rotation by seed
        k = seed % len(base)
        return base[k:] + base[:k]
    return list(base)


def _map_color_declaration(match: re.Match, seed: int) -> str:
    prop = match.group(1)
    value = match.group(2)
    index = {"i": seed % 11}

    def repl(token: re.Match) -> str:
        index["i"] += 1
        return _map_css_color(prop, token.group(0), index["i"])

    return f"{prop}: {_RE_COLOR_TOKEN.sub(repl, value)}"


def _map_css_color(prop: str, raw_color: str, index: int) -> str:
    rgb = _parse_css_rgb(raw_color)
    if rgb is None:
        return raw_color
    luminance = _relative_luminance(rgb)
    saturation = _rgb_saturation(rgb)
    role = _property_role(prop)

    if role == "background":
        if luminance >= 0.72:
            palette = ("#FFFFFF", "#F8FAFC", "#F1F5F9", "#E2E8F0")
        elif luminance <= 0.32:
            palette = ("#0F172A", "#111827", "#1E293B", "#172033")
        elif saturation >= 0.30:
            palette = ("#1D4ED8", "#047857", "#0F766E", "#0369A1")
        else:
            palette = ("#334155", "#475569", "#64748B")
    elif role == "border":
        if luminance >= 0.55:
            palette = ("#CBD5E1", "#D1D5DB", "#E2E8F0")
        else:
            palette = ("#334155", "#475569", "#1F2937")
    else:
        if luminance >= 0.72:
            palette = ("#FFFFFF", "#F8FAFC", "#E2E8F0")
        elif luminance <= 0.34:
            palette = ("#0F172A", "#111827", "#1E293B")
        elif saturation >= 0.30:
            palette = ("#1D4ED8", "#047857", "#0F766E")
        else:
            palette = ("#334155", "#475569", "#64748B")
    return palette[index % len(palette)]


def _property_role(prop: str) -> str:
    p = prop.lower()
    if "background" in p or p in {"fill"} or "--bg" in p or "--background" in p:
        return "background"
    if "border" in p or "shadow" in p or "--border" in p:
        return "border"
    return "text"


def _parse_css_rgb(raw: str) -> tuple[int, int, int] | None:
    value = raw.strip().lower()
    if value.startswith("#"):
        h = value.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) not in (6, 8):
            return None
        try:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        except ValueError:
            return None
    if value.startswith("rgb"):
        nums = re.findall(r"[\d.]+", value)
        if len(nums) < 3:
            return None
        try:
            return tuple(max(0, min(255, int(float(num)))) for num in nums[:3])  # type: ignore[return-value]
        except ValueError:
            return None
    return None


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    vals = []
    for channel in rgb:
        c = channel / 255.0
        vals.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]


def _rgb_saturation(rgb: tuple[int, int, int]) -> float:
    vals = [channel / 255.0 for channel in rgb]
    hi, lo = max(vals), min(vals)
    return 0.0 if hi <= 0 else (hi - lo) / hi
