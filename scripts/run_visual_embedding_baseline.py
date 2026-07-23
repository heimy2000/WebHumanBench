#!/usr/bin/env python3
"""Extract frozen UIClip image embeddings and run one-class visual baselines."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webmark.release import canonical_json_sha256, sha256_file
from webmark.visual_embedding_baseline import evaluate_visual_embedding_baselines

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmark" / "webhumanbench_v1" / "webhumanbench_manifest.json"
DEFAULT_CAPTURE_LEDGER = ROOT / "benchmark" / "webhumanbench_v1" / "capture_ledger.json"
DEFAULT_EMBEDDINGS = ROOT / "results" / "webhumanbench_v1_uiclip_image_embeddings.json"
DEFAULT_OUTPUT = ROOT / "results" / "webhumanbench_v1_visual_embedding_baselines.json"
DEFAULT_MODEL_ID = "biglab/uiclip_jitteredwebsites-2-224-paraphrased"
DEFAULT_MODEL_REVISION = "fec2aa208f96d831f430fd8f13cd9d5fa41bdef7"
DEFAULT_PROCESSOR_ID = "openai/clip-vit-base-patch32"
DEFAULT_PROCESSOR_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DEFAULT_MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--biglab--uiclip_jitteredwebsites-2-224-paraphrased"
    / "snapshots"
    / DEFAULT_MODEL_REVISION
)
DEFAULT_PROCESSOR_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32"
    / "snapshots"
    / DEFAULT_PROCESSOR_REVISION
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _recorded_path(path: Path) -> str:
    """Return a stable repository-relative path when possible.

    Command-line paths are commonly relative to the caller's working
    directory. Resolve them before comparing with ROOT so the documented
    reproduction commands work as written.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _screenshot_rows(
    manifest: dict[str, Any],
    capture_ledger: dict[str, Any],
    *,
    capture_root: Path,
    verify_files: bool,
) -> list[dict[str, Any]]:
    historical = {str(row["id"]): row for row in capture_ledger.get("records", [])}
    rows = []
    seen: set[str] = set()
    for record in manifest.get("records", []):
        group_id = str(record["group_id"])
        if group_id in seen:
            raise ValueError(f"manifest has multiple scoring screenshots for group {group_id}")
        seen.add(group_id)
        if record["source"] == "human":
            capture_id = str(record["provenance"]["capture_id"])
            if capture_id not in historical:
                raise ValueError(f"capture ledger is missing {capture_id}")
            artifact = historical[capture_id]["artifacts"]["screenshot"]
        else:
            artifact = record["provenance"]["artifacts"]["screenshot"]
        path = capture_root / str(artifact["path"])
        digest = str(artifact["sha256"])
        if verify_files:
            if not path.is_file():
                raise ValueError(f"screenshot is missing: {path}")
            digest = sha256_file(path)
            if digest != artifact["sha256"]:
                raise ValueError(f"screenshot hash mismatch: {path}")
        rows.append(
            {
                "id": str(record["id"]),
                "group_id": group_id,
                "source": str(record["source"]),
                "split": str(record["split"]),
                "page_type": str(record["page_type"]),
                "model_id": record.get("model_id"),
                "screenshot_path": _recorded_path(path),
                "screenshot_sha256": digest,
            }
        )
    return sorted(rows, key=lambda row: row["group_id"])


def _embedding_payload(
    rows: list[dict[str, Any]],
    *,
    model_path: Path,
    processor_path: Path,
    model_id: str,
    model_revision: str,
    processor_id: str,
    processor_revision: str,
    batch_size: int,
    device_name: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import CLIPImageProcessorPil, CLIPModel
    except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
        raise RuntimeError("Install the optional vision dependencies: torch transformers pillow") from exc
    if not model_path.is_dir() or not processor_path.is_dir():
        raise ValueError("the frozen UIClip model and processor snapshots must exist locally")
    weights = model_path / "model.safetensors"
    model_config = model_path / "config.json"
    processor_config = processor_path / "preprocessor_config.json"
    for path in (weights, model_config, processor_config):
        if not path.is_file():
            raise ValueError(f"required model artifact is missing: {path}")

    model = CLIPModel.from_pretrained(model_path, local_files_only=True).eval()
    processor = CLIPImageProcessorPil.from_pretrained(processor_path, local_files_only=True)
    device = torch.device(device_name)
    model.to(device)
    embedded = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = []
        try:
            for row in batch:
                images.append(Image.open(ROOT / row["screenshot_path"]).convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.inference_mode():
                pooled = model.vision_model(pixel_values=pixel_values).pooler_output
                vectors = model.visual_projection(pooled)
                vectors = vectors / vectors.norm(dim=-1, keepdim=True)
            arrays = vectors.detach().cpu().to(torch.float32).tolist()
        finally:
            for image in images:
                image.close()
        for row, vector in zip(batch, arrays, strict=True):
            embedded.append(
                {
                    **row,
                    "embedding": vector,
                    "embedding_sha256": canonical_json_sha256(vector),
                }
            )
        print(f"UIClip embedding [{len(embedded)}/{len(rows)}]")
    return {
        "schema": "webmark_uiclip_image_embeddings_v1",
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "model": {
            "model_id": model_id,
            "revision": model_revision,
            "model_config_sha256": sha256_file(model_config),
            "model_weights_sha256": sha256_file(weights),
            "processor_id": processor_id,
            "processor_revision": processor_revision,
            "processor_config_sha256": sha256_file(processor_config),
            "embedding": "l2_normalized_visual_projection_output",
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "device": str(device),
        },
        "summary": {
            "n_groups": len(embedded),
            "embedding_dimension": len(embedded[0]["embedding"]),
            "batch_size": batch_size,
        },
        "records": embedded,
    }


def _validate_embedding_cache(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    manifest_sha256: str,
) -> dict[str, list[float]]:
    if payload.get("schema") != "webmark_uiclip_image_embeddings_v1":
        raise ValueError("embedding cache has an unsupported schema")
    if payload.get("status") != "complete" or payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("embedding cache is incomplete or belongs to another manifest")
    expected = {row["group_id"]: row for row in rows}
    observed = {str(row.get("group_id")): row for row in payload.get("records", [])}
    if set(expected) != set(observed):
        raise ValueError("embedding cache group IDs do not match the manifest")
    embeddings = {}
    for group_id, expected_row in expected.items():
        row = observed[group_id]
        if row.get("screenshot_sha256") != expected_row["screenshot_sha256"]:
            raise ValueError(f"embedding screenshot hash changed for {group_id}")
        for key in ("id", "source", "split", "page_type", "model_id"):
            if row.get(key) != expected_row.get(key):
                raise ValueError(f"embedding metadata changed for {group_id}: {key}")
        vector = row.get("embedding")
        if not isinstance(vector, list) or row.get("embedding_sha256") != canonical_json_sha256(vector):
            raise ValueError(f"embedding hash failed for {group_id}")
        embeddings[group_id] = [float(value) for value in vector]
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--capture-ledger", type=Path, default=DEFAULT_CAPTURE_LEDGER)
    parser.add_argument("--embedding-output", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--processor-path", type=Path, default=DEFAULT_PROCESSOR_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--processor-id", default=DEFAULT_PROCESSOR_ID)
    parser.add_argument("--processor-revision", default=DEFAULT_PROCESSOR_REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.n_bootstrap <= 0 or args.n_splits <= 0:
        raise ValueError("batch, bootstrap, and split counts must be positive")

    manifest = _load_json(args.manifest)
    capture_ledger = _load_json(args.capture_ledger)
    capture_root = args.manifest.parent / "captures"
    screenshot_rows = _screenshot_rows(
        manifest,
        capture_ledger,
        capture_root=capture_root,
        # Analysis releases intentionally omit restricted screenshots. In that
        # mode, validate the published screenshot digests bound to the cached
        # embeddings instead of pretending to recompute unavailable bytes.
        verify_files=not args.analysis_only,
    )
    manifest_sha256 = canonical_json_sha256(manifest)
    if args.analysis_only:
        embedding_payload = _load_json(args.embedding_output)
    else:
        embedding_payload = _embedding_payload(
            screenshot_rows,
            model_path=args.model_path,
            processor_path=args.processor_path,
            model_id=args.model_id,
            model_revision=args.model_revision,
            processor_id=args.processor_id,
            processor_revision=args.processor_revision,
            batch_size=args.batch_size,
            device_name=args.device,
            manifest_sha256=manifest_sha256,
        )
        _write_json(args.embedding_output, embedding_payload)
    embeddings = _validate_embedding_cache(
        embedding_payload,
        screenshot_rows,
        manifest_sha256=manifest_sha256,
    )
    analysis = evaluate_visual_embedding_baselines(
        manifest,
        embeddings,
        n_resamples=args.n_bootstrap,
        n_alternative_splits=args.n_splits,
    )
    result = {
        **analysis,
        "inputs": {
            "manifest": {"path": _recorded_path(args.manifest), "sha256": sha256_file(args.manifest)},
            "capture_ledger": {
                "path": _recorded_path(args.capture_ledger),
                "sha256": sha256_file(args.capture_ledger),
            },
            "embedding_cache": {
                "path": _recorded_path(args.embedding_output),
                "sha256": sha256_file(args.embedding_output),
            },
        },
        "encoder": embedding_payload["model"],
        "analysis_profile": {
            "mode": "cached_embeddings_only" if args.analysis_only else "fresh_embedding_extraction",
            "raw_screenshot_bytes_verified": not args.analysis_only,
        },
    }
    _write_json(args.output, result)
    print("visual embedding baselines")
    for name, baseline in result["baselines"].items():
        frozen = baseline["frozen_type_macro"]
        resplit = baseline["source_split_sensitivity"]
        loro = baseline["leave_one_reference_out"]["summary"]
        print(
            f"  {name}: frozen={frozen['point_estimate']:.4f}; "
            f"resplit median={resplit['median']:.4f}; LORO={loro['mean']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
