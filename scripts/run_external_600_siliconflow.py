#!/usr/bin/env python3
"""SE-2 external-API 600-page execution audit via SiliconFlow.

This is the network-backed counterpart to ``run_synthetic_600_stress.py``.
It calls the SiliconFlow OpenAI-compatible chat-completions endpoint, caches
every raw response and generated HTML file, then runs the historical
four-operator, repeat-allowed K=3/L=5 controller used to create the completed
SE-2C artifact. The logged ``early_stop_z`` field in that artifact was
inactive: the controller did not implement branch termination. The current
three-operator guarded/no-repeat release setting is evaluated separately by
SE-26 and SE-27.

The script is intentionally resumable. A 600-page run can be interrupted and
continued without re-calling pages whose cached HTML already exists. Explicit
page IDs may be selected for provenance-preserving retries of rejected rows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from webmark.beam_search import BeamSearchConfig, beam_search_correct
from webmark.bias import BiasScorer, ReferenceStats
from webmark.features import extract_page_features
from webmark.operators import OperatorName, apply_operator

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "external_600_siliconflow.json"
DEFAULT_CACHE_DIR = ROOT / "results" / "external_600_siliconflow_cache"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
PAGE_TYPES = (
    "saas_landing",
    "docs_homepage",
    "product_showcase",
    "developer_tool",
    "dashboard_shell",
    "portfolio_showcase",
)
DEFAULT_MODELS = (
    "Qwen/Qwen3-32B",
    "deepseek-ai/DeepSeek-V4-Flash",
    "MiniMaxAI/MiniMax-M2.5",
    "tencent/Hunyuan-A13B-Instruct",
    "zai-org/GLM-5.2",
    "Pro/moonshotai/Kimi-K2.6",
    "meituan-longcat/LongCat-2.0",
)
TYPOGRAPHY_REF = [12.0, 14.0, 16.0, 18.0, 19.0, 19.5, 20.0, 21.0, 22.0, 24.0, 26.0, 28.0]
HISTORICAL_SE2C_OPERATORS: tuple[OperatorName, ...] = (
    "font_scale",
    "spacing",
    "radius",
    "color",
)


PAGE_TYPE_BRIEFS: dict[str, str] = {
    "saas_landing": (
        "a SaaS landing page with a hero, feature cards, pricing teaser, testimonials, "
        "and a final call-to-action"
    ),
    "docs_homepage": (
        "a documentation homepage with a product intro, quick-start steps, doc cards, "
        "API reference links, and a search affordance"
    ),
    "product_showcase": (
        "a product showcase page with narrative hero copy, product highlights, gallery "
        "cards, specifications, and a purchase call-to-action"
    ),
    "developer_tool": (
        "a developer-tool homepage with terminal/code snippets, integration cards, "
        "workflow steps, and docs/GitHub actions"
    ),
    "dashboard_shell": (
        "a dashboard shell with sidebar navigation, metric cards, table/list content, "
        "filters, and compact controls"
    ),
    "portfolio_showcase": (
        "a creative portfolio page with project tiles, biography section, case-study "
        "links, and contact call-to-action"
    ),
}


SCENARIOS = (
    "privacy-first analytics",
    "AI coding assistant",
    "open-source design system",
    "climate-data platform",
    "collaborative notes workspace",
    "robotics monitoring console",
    "medical scheduling tool",
    "fintech reconciliation app",
    "education course builder",
    "creator-commerce storefront",
    "security observability product",
    "research lab homepage",
)


@dataclass(frozen=True)
class PlannedPage:
    page_id: str
    page_type: str
    model: str
    index: int
    seed: int
    scenario: str


def _reference() -> ReferenceStats:
    return ReferenceStats(
        means={"typography": 19.0, "spacing": 1.5, "grid": 5.0, "color": 5.0, "saturation": 0.4},
        stds={"typography": 5.0, "spacing": 0.3, "grid": 2.0, "color": 2.0, "saturation": 0.1},
        wasserstein_samples_per_dim={"typography": TYPOGRAPHY_REF},
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def _load_env_file(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _api_key(env_name: str) -> str:
    value = os.environ.get(env_name) or os.environ.get("SF_API_KEY")
    if not value:
        raise SystemExit(
            f"Missing SiliconFlow API key. Set {env_name}=sk-... "
            "or put it in code/.env before running the external execution audit."
        )
    return value


def _planned_pages(models: Sequence[str], pages_per_type: int, seed: int) -> list[PlannedPage]:
    pages: list[PlannedPage] = []
    cursor = 0
    for type_idx, page_type in enumerate(PAGE_TYPES):
        for i in range(pages_per_type):
            model = models[cursor % len(models)]
            rng = random.Random(seed + type_idx * 10_000 + i)
            scenario = SCENARIOS[rng.randrange(len(SCENARIOS))]
            pages.append(
                PlannedPage(
                    page_id=f"sf_{page_type}_{i:03d}",
                    page_type=page_type,
                    model=model,
                    index=i,
                    seed=seed + type_idx * 10_000 + i,
                    scenario=scenario,
                )
            )
            cursor += 1
    return pages


def _select_planned_pages(
    pages: Sequence[PlannedPage],
    page_ids: Sequence[str] | None,
    max_pages: int | None,
) -> list[PlannedPage]:
    """Select explicit retry IDs without changing their original plan metadata."""
    selected = list(pages)
    if page_ids:
        requested = [str(page_id) for page_id in page_ids]
        if len(requested) != len(set(requested)):
            raise ValueError("--page-id values must be unique")
        known = {page.page_id for page in selected}
        unknown = sorted(set(requested).difference(known))
        if unknown:
            raise ValueError(f"unknown --page-id values: {', '.join(unknown[:3])}")
        requested_set = set(requested)
        selected = [page for page in selected if page.page_id in requested_set]
    if max_pages is not None:
        if max_pages <= 0:
            raise ValueError("--max-pages must be positive")
        selected = selected[:max_pages]
    return selected


def _prompt(page: PlannedPage) -> list[dict[str, str]]:
    brief = PAGE_TYPE_BRIEFS[page.page_type]
    system = (
        "You generate self-contained, production-like HTML/CSS webpages for a research "
        "benchmark. Return only one complete HTML document. Do not wrap the answer in "
        "Markdown fences and do not include explanations."
    )
    user = f"""
Create {brief}.

Benchmark constraints:
- Scenario/theme: {page.scenario}
- Page type label: {page.page_type}
- Output exactly one complete HTML document with <!DOCTYPE html>, <html lang="en">,
  <head>, <title>, inline <style>, and <body>.
- No external assets, scripts, icon libraries, web fonts, iframes, or network calls.
- Use semantic landmarks where natural: header, nav, main, section, article, footer.
- Use literal CSS values, including px font-size declarations, line-height values,
  border-radius, margin/padding/gap, and literal hex colors. Avoid CSS variables for
  primary color/font tokens so the statistical extractor can read the page.
- Include at least one interactive element such as links or buttons and make every
  image-like placeholder text-based or include alt text.
- Keep the page between 500 and 1200 words of HTML/CSS source.
- The design should be plausible for a modern human-authored website, but do not
  overfit to any specific brand.

Variation seed: {page.seed}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_html(content: str) -> str:
    text = content.strip()
    fence = re.search(r"```(?:html)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    doc = re.search(r"(?is)(<!doctype\s+html\b.*?</html\s*>)", text)
    if doc:
        return doc.group(1).strip()
    html = re.search(r"(?is)(<html\b.*?</html\s*>)", text)
    if html:
        return "<!DOCTYPE html>\n" + html.group(1).strip()
    raise ValueError("model response did not contain a complete <html>...</html> document")


def _post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    retries: int,
    retry_sleep_s: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if model.startswith(("Qwen/Qwen3", "deepseek-ai/DeepSeek", "zai-org/GLM", "Pro/")):
        payload["enable_thinking"] = False

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "webmark-aaai-se2/1.0",
    }

    last_error = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
                response_headers = {k.lower(): v for k, v in resp.headers.items()}
                return json.loads(body), response_headers
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:1000]}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            if attempt >= retries:
                break
        sleep_for = retry_sleep_s * (2 ** attempt) + random.random() * 0.2
        time.sleep(sleep_for)
    raise RuntimeError(last_error or "unknown SiliconFlow request failure")


def _apply_chain(html: str, chain: Sequence[str]) -> str:
    out = html
    for op in chain:
        out = apply_operator(out, op)
    return out


def _summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("status") == "ok" and math.isfinite(float(r["bias_delta"]))]
    deltas = [float(r["bias_delta"]) for r in valid]
    neg = sum(1 for d in deltas if d < 0)
    pos = sum(1 for d in deltas if d > 0)
    zero = len(deltas) - neg - pos
    return {
        "n_rows": len(rows),
        "n_valid": len(valid),
        "negative_n": neg,
        "positive_n": pos,
        "zero_n": zero,
        "mean_delta": statistics.fmean(deltas) if deltas else float("nan"),
        "median_delta": statistics.median(deltas) if deltas else float("nan"),
        "std_delta": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "mean_chain_length": (
            statistics.fmean(len(r["operator_chain"]) for r in valid) if valid else float("nan")
        ),
    }


def _feature_summary(html: str) -> dict[str, Any]:
    feats = extract_page_features(html)
    return feats.to_summary()


def _write_json(path: pathlib.Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _relative(path: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--env-file", type=pathlib.Path, default=ROOT / ".env")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", action="append", default=None, help="SiliconFlow model string; repeat to use multiple models")
    parser.add_argument("--pages-per-type", type=int, default=100)
    parser.add_argument(
        "--page-id",
        action="append",
        help="Run only this canonical planned page ID; repeat for a retry subset",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Optional smoke-test cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep-s", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Only write the planned run manifest; no API calls")
    args = parser.parse_args()

    _load_env_file(args.env_file)
    models = tuple(args.model or DEFAULT_MODELS)
    pages = _select_planned_pages(
        _planned_pages(models, args.pages_per_type, args.seed),
        args.page_id,
        args.max_pages,
    )

    manifest = {
        "experiment": "SE-2 external-API 600-page execution audit via SiliconFlow",
        "scope": (
            "Network-backed external-model execution audit. Generated pages are cached and "
            "then evaluated by the historical SE-2C WebMark feature/scorer/operator/"
            "beam stack. Current guarded/no-repeat re-scoring is reported in SE-26/SE-27. "
            "The zero-change fallback censors scorer movement, so delta summaries are descriptive only."
        ),
        "config": {
            "base_url": args.base_url,
            "models": list(models),
            "page_types": list(PAGE_TYPES),
            "pages_per_type": args.pages_per_type,
            "planned_pages": len(pages),
            "selected_page_ids": [page.page_id for page in pages] if args.page_id else None,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "operators": list(HISTORICAL_SE2C_OPERATORS),
            "beam_width": 3,
            "max_depth": 5,
            "allow_operator_reuse": True,
            "enforce_contrast_guardrail": False,
            "cache_dir": _relative(args.cache_dir),
        },
        "planned_pages": [page.__dict__ for page in pages],
    }
    if args.dry_run:
        _write_json(args.output, {**manifest, "status": "dry_run"})
        print(f"Dry run only. Planned {len(pages)} pages; wrote {args.output}")
        return 0

    api_key = _api_key(args.api_key_env)
    html_dir = args.cache_dir / "html"
    raw_dir = args.cache_dir / "raw"
    html_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    scorer = BiasScorer(_reference(), penalty="l2", nonparametric_dims=("typography",))
    cfg = BeamSearchConfig(
        beam_width=3,
        max_depth=5,
        operators=HISTORICAL_SE2C_OPERATORS,
        seed=args.seed,
        allow_operator_reuse=True,
        enforce_contrast_guardrail=False,
    )

    rows: list[dict[str, Any]] = []
    usage_totals: dict[str, int] = defaultdict(int)
    start = time.time()

    for n, page in enumerate(pages, start=1):
        initial_path = html_dir / f"{page.page_id}_initial.html"
        corrected_path = html_dir / f"{page.page_id}_corrected.html"
        raw_path = raw_dir / f"{page.page_id}.json"
        row_base = {
            "id": page.page_id,
            "page_type": page.page_type,
            "model": page.model,
            "index": page.index,
            "seed": page.seed,
            "scenario": page.scenario,
            "initial_html_path": _relative(initial_path),
            "corrected_html_path": _relative(corrected_path),
            "raw_response_path": _relative(raw_path),
        }

        try:
            if initial_path.exists() and raw_path.exists():
                html = initial_path.read_text(encoding="utf-8")
                response = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                response, headers = _post_chat_completion(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=page.model,
                    messages=_prompt(page),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                    retries=args.retries,
                    retry_sleep_s=args.retry_sleep_s,
                )
                response["_webmark_headers"] = {
                    "x-siliconcloud-trace-id": headers.get("x-siliconcloud-trace-id", "")
                }
                raw_path.write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")
                content = response["choices"][0]["message"]["content"]
                html = _extract_html(content)
                initial_path.write_text(html, encoding="utf-8")

            for key, value in (response.get("usage") or {}).items():
                if isinstance(value, int):
                    usage_totals[key] += value

            feats_initial = extract_page_features(html)
            bias_initial = scorer.score(feats_initial).total
            chain, _ = beam_search_correct(html, scorer, config=cfg, initial_features=feats_initial)
            corrected_html = _apply_chain(html, chain)
            corrected_path.write_text(corrected_html, encoding="utf-8")
            feats_corrected = extract_page_features(corrected_html)
            bias_corrected = scorer.score(feats_corrected).total

            rows.append(
                {
                    **row_base,
                    "status": "ok",
                    "operator_chain": list(chain),
                    "bias_initial": bias_initial,
                    "bias_corrected": bias_corrected,
                    "bias_delta": bias_corrected - bias_initial,
                    "features_initial": feats_initial.to_summary(),
                    "features_corrected": feats_corrected.to_summary(),
                    "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                    "usage": response.get("usage", {}),
                    "trace_id": response.get("_webmark_headers", {}).get("x-siliconcloud-trace-id", ""),
                }
            )
        except Exception as exc:
            rows.append({**row_base, "status": "failed", "error": repr(exc)})

        if n % max(1, args.save_every) == 0 or n == len(pages):
            out = _build_output(manifest, rows, usage_totals, start)
            _write_json(args.output, out)
            ok_n = sum(1 for r in rows if r.get("status") == "ok")
            print(f"[{n}/{len(pages)}] ok={ok_n} failed={len(rows)-ok_n} output={args.output}")

    out = _build_output(manifest, rows, usage_totals, start)
    _write_json(args.output, out)
    summary = out["summary"]
    print("SE-2 external-API execution audit via SiliconFlow")
    print(f"  rows planned: {summary['n_rows']}")
    print(f"  valid deltas: {summary['n_valid']}")
    print(f"  negative/positive/zero: {summary['negative_n']}/{summary['positive_n']}/{summary['zero_n']}")
    print(f"  mean delta: {summary['mean_delta']:.3f}")
    return 0


def _build_output(
    manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    usage_totals: dict[str, int],
    start_time: float,
) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["page_type"]].append(row)
        by_model[row["model"]].append(row)

    summary = _summarise(rows)
    failed = [r for r in rows if r.get("status") != "ok"]
    return {
        **{k: v for k, v in manifest.items() if k != "planned_pages"},
        "status": "complete" if len(rows) == manifest["config"]["planned_pages"] else "partial",
        "summary": {
            **summary,
            "planned_pages": manifest["config"]["planned_pages"],
            "failed_n": len(failed),
            "elapsed_s": time.time() - start_time,
            "usage_totals": dict(usage_totals),
        },
        "by_page_type": {k: _summarise(v) for k, v in sorted(by_type.items())},
        "by_model": {k: _summarise(v) for k, v in sorted(by_model.items())},
        "failures": failed[:50],
        "per_page": list(rows),
    }


if __name__ == "__main__":
    sys.exit(main())
