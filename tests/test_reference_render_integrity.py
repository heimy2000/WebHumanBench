"""Tests for deterministic capture render-integrity screening."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_reference_render_integrity.py"
    spec = importlib.util.spec_from_file_location("audit_reference_render_integrity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visible_text_excludes_script_and_style_content() -> None:
    module = _load_module()
    html = "<style>hidden words</style><main>Visible text</main><script>secret words</script>"

    assert module._visible_text(html) == "Visible text"


def test_screen_record_rejects_near_uniform_or_error_like_capture(tmp_path: Path) -> None:
    module = _load_module()
    html_path = tmp_path / "page.html"
    screenshot_path = tmp_path / "page.png"
    html_path.write_text("<main>Application error {{ .Site.Title }}</main>", encoding="utf-8")
    Image.new("RGB", (8, 8), "white").save(screenshot_path)
    record = {
        "id": "source@390x844",
        "source_id": "source",
        "viewport": "390x844",
        "capture_html_sha256": module.sha256_file(html_path),
        "features": {
            "typography": [16, 18],
            "spacing": [1.2, 1.4],
            "grid": [0, 1],
            "color": ["rgb(0, 0, 0)", "rgb(255, 255, 255)"],
            "saturation": [0, 0],
        },
        "artifacts": {
            "html": {"path": "page.html", "sha256": module.sha256_file(html_path)},
            "screenshot": {"path": "page.png", "sha256": module.sha256_file(screenshot_path)},
        },
    }

    result = module._screen_record(record, tmp_path)

    assert result["status"] == "fail"
    assert "visible_text_below_minimum" in result["reasons"]
    assert "error_marker:application error" in result["reasons"]
    assert "unrendered_template_syntax:{{ ." in result["reasons"]
    assert "near_uniform_screenshot" in result["reasons"]
