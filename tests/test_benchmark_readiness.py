"""Tests for the formal WebHumanBench corpus-coverage gate."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_human_likeness import _manifest

from webmark.benchmark_readiness import audit_benchmark_readiness


def test_small_fixture_is_explicitly_pilot_only() -> None:
    result = audit_benchmark_readiness(_manifest())
    assert result["status"] == "pilot_only"
    assert result["formal_historical_source_target"] == 1200
    assert result["findings"]
