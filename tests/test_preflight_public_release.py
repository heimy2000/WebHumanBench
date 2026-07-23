"""Tests for the public release staging-tree preflight."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_preflight():
    path = Path(__file__).resolve().parents[1] / "scripts" / "preflight_public_release.py"
    spec = importlib.util.spec_from_file_location("preflight_public_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_accepts_clean_staging_tree(tmp_path: Path):
    preflight = _load_preflight()
    (tmp_path / "README.md").write_text("release notes", encoding="utf-8")

    report = preflight.preflight_public_release(tmp_path, max_file_size_bytes=100)

    assert report["status"] == "pass"
    assert report["issues"] == []


def test_preflight_detects_local_metadata_dependency_and_token(tmp_path: Path):
    preflight = _load_preflight()
    (tmp_path / "._README.md").write_text("metadata", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "webmark.egg-info").mkdir()
    (tmp_path / "token.txt").write_text("prefix " + "sk-" + "x" * 24, encoding="utf-8")
    (tmp_path / "large.bin").write_bytes(b"123")

    report = preflight.preflight_public_release(tmp_path, max_file_size_bytes=2)

    findings = {(issue["kind"], issue["path"]) for issue in report["issues"]}
    assert report["status"] == "fail"
    assert ("os_metadata", "._README.md") in findings
    assert ("forbidden_directory", "node_modules") in findings
    assert ("forbidden_directory", "webmark.egg-info") in findings
    assert ("credential_pattern", "token.txt") in findings
    assert ("oversized_file", "large.bin") in findings
