"""CSS-level feature extraction for the five-dimensional bias vector.

The five dimensions mirror §3.1 of the paper:

1. ``typography`` — supported CSS font-size samples converted to px.
2. ``spacing``    — literal line-height values.
3. ``grid``       — an inline margin-left modulo-12 proxy.
4. ``color``      — literal foreground/background color tokens, reduced to
   unique-token cardinality by the scorer.
5. ``saturation`` — mean saturation across parsed literal colors.

All extraction is deterministic and offline: every input is scanned for
supported literal declarations in an HTML/CSS source string, with no browser
computed-style pass or LLM calls. We expose a single ``extract_page_features``
function plus a thin ``PageFeatures`` dataclass so downstream code does not
need to thread five separate dicts through every call.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

# ── dimension names: must match bias.ReferenceStats ────────────────────────────
FEATURE_NAMES: tuple[str, ...] = (
    "typography",
    "spacing",
    "grid",
    "color",
    "saturation",
)

# ── regex constants (kept module-level so they compile once) ───────────────────
_RE_FONT_PX = re.compile(r"font-size\s*:\s*([\d.]+)\s*px", re.IGNORECASE)
_RE_FONT_EM = re.compile(r"font-size\s*:\s*([\d.]+)\s*em", re.IGNORECASE)
_RE_LH = re.compile(r"line-height\s*:\s*([\d.]+)", re.IGNORECASE)
_RE_COLOR = re.compile(r"color\s*:\s*(#[0-9a-fA-F]{3,8}|rgba?\([^)]+\))", re.IGNORECASE)
_RE_BG = re.compile(r"background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,8}|rgba?\([^)]+\))", re.IGNORECASE)
_RE_BODY_PX = re.compile(r"body\s*\{[^}]*font-size\s*:\s*([\d.]+)\s*px", re.IGNORECASE)
_RE_H1_PX = re.compile(r"<h1[^>]*style\s*=\s*\"[^\"]*font-size\s*:\s*([\d.]+)\s*px", re.IGNORECASE)


@dataclass
class PageFeatures:
    """Five-dimensional CSS-level feature vector for a single page."""

    typography: list[float] = field(default_factory=list)
    spacing: list[float] = field(default_factory=list)
    grid: list[float] = field(default_factory=list)
    color: list[str] = field(default_factory=list)
    saturation: list[float] = field(default_factory=list)

    def to_summary(self) -> dict[str, float]:
        """Reduce each dimension to a single summary statistic for caching."""

        def _mean(xs: Sequence[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        def _std(xs: Sequence[float]) -> float:
            if len(xs) < 2:
                return 0.0
            mu = _mean(xs)
            return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))

        return {
            "typography_mean": _mean(self.typography),
            "typography_std": _std(self.typography),
            "typography_n": len(self.typography),
            "spacing_mean": _mean(self.spacing),
            "spacing_std": _std(self.spacing),
            "spacing_n": len(self.spacing),
            "grid_mean": _mean(self.grid),
            "grid_std": _std(self.grid),
            "grid_n": len(self.grid),
            "color_unique": len({c.lower() for c in self.color if c}),
            "color_n": len(self.color),
            "saturation_mean": _mean(self.saturation),
            "saturation_std": _std(self.saturation),
            "saturation_n": len(self.saturation),
        }

    def asdict(self) -> dict[str, list[float]]:
        return asdict(self)


def extract_page_features(html: str) -> PageFeatures:
    """Extract the five-dimensional feature vector from one HTML/CSS string.

    The function is deterministic and pure: identical HTML strings produce
    identical ``PageFeatures`` outputs. This invariance is what makes the
    bias score reproducible across machines.

    Parameters
    ----------
    html : str
        HTML with supported literal CSS declarations. The function tolerates
        partial or malformed HTML because it has no parser dependency.

    Returns
    -------
    PageFeatures with five populated lists.
    """

    pf = PageFeatures()

    # ── 1. typography (raw px measurements; downstream bias uses W-1) ──
    px_values: list[float] = []
    px_values.extend(float(m) for m in _RE_FONT_PX.findall(html))
    em_values: list[float] = []
    em_values.extend(float(m) for m in _RE_FONT_EM.findall(html))
    body_px_match = _RE_BODY_PX.search(html)
    body_px = float(body_px_match.group(1)) if body_px_match else 16.0
    if em_values:
        # Body-scaled: if a body font-size is set, scale em values.
        px_values.extend(e * body_px for e in em_values)
    h1_match = _RE_H1_PX.search(html)
    if h1_match:
        px_values.append(float(h1_match.group(1)))
    pf.typography = px_values

    # ── 2. spacing (line-height, normalised by font-size when available) ──
    lh_values = [float(m) for m in _RE_LH.findall(html)]
    pf.spacing = lh_values if lh_values else []

    # ── 3. grid (inline margin-left modulo-12 proxy) ──
    grid_values: list[float] = []
    for block_match in re.finditer(
        r"<(?:section|div|main|article)[^>]*style=\"[^\"]*margin-left\s*:\s*([\d.]+)\s*px",
        html, re.IGNORECASE,
    ):
        v = float(block_match.group(1))
        if v > 0:
            grid_values.append(round((v / 8.0) % 12, 3))
    pf.grid = grid_values

    # ── 4. colour (palette list, lowercased) ──
    pf.color = [c.lower() for c in _RE_COLOR.findall(html)] + [c.lower() for c in _RE_BG.findall(html)]
    if not pf.color:
        pf.color = []

    # ── 5. saturation (HSV-S, mean of all palette colours) ──
    sat_values: list[float] = []
    for c in pf.color:
        s = _hsv_s_from_css(c)
        if s is not None:
            sat_values.append(s)
    pf.saturation = sat_values

    return pf


# ── internal helpers ─────────────────────────────────────────────────────────
def _hsv_s_from_css(color: str) -> float | None:
    """Return HSV-S in [0, 1] for a CSS color string, or None if unparsable."""

    c = color.strip().lower()
    if c.startswith("#"):
        h = c.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) not in (6, 8):
            return None
        try:
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
        except ValueError:
            return None
    elif c.startswith("rgb"):
        nums = re.findall(r"[\d.]+", c)
        if len(nums) < 3:
            return None
        try:
            r, g, b = (float(nums[i]) / 255.0 for i in range(3))
        except ValueError:
            return None
    else:
        return None
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx <= 0 or mx == mn:
        return 0.0
    return (mx - mn) / mx


def aggregate_typography_summary(pf: PageFeatures) -> list[float]:
    """Return the typography dimension as a single quantile-summary list.

    Used by ``bias.wasserstein1_distance`` on the typography dimension
    (paper §3.3 — SE-6 partial non-parametric replacement).
    """

    if not pf.typography:
        return []
    xs = sorted(pf.typography)
    n = len(xs)
    qs = [xs[min(n - 1, int(round(q * (n - 1))))] for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    return qs
