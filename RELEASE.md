# Anonymous Analysis Release: WebHumanBench v1.0.0-rc5

This is the push-ready anonymous release package for the submission
*WebHumanBench: An Auditable Benchmark Pilot for Reference-Relative Web Design
Fit*. It is an **analysis-only** release profile, not a claim that every raw
artifact can be redistributed.

## What This Package Contains

- Apache-2.0 implementation, release scripts, and release-facing tests.
- Versioned source, capture, split, and generation ledgers with canonical
  hashes, plus the feature-bearing benchmark manifest.
- Frozen baseline, sensitivity, prompt, protocol, browser-version, and
  image-control results.
- A data card, third-party notices, reproducibility instructions, and a
  machine-readable `RELEASE_MANIFEST.json` generated during staging.

The manifest-only integrity audit and the five frozen reference-fit baselines
run offline from this package. A passing audit establishes internal consistency
of the released ledgers and feature-bearing benchmark manifest; it does not
establish authorship, preference, accessibility, quality, or a universal human
design construct.

## Intentionally Excluded Materials

The package does **not** distribute source captures, vendored remote assets,
snapshot overlays, generated HTML, raw provider responses, or screenshots.
Their paths and SHA-256 values remain recorded in the included ledgers for
provenance. They are excluded because they may remain subject to upstream,
asset-owner, or provider terms. See `THIRD_PARTY_ARTIFACT_NOTICES.md`.

## Pre-Push Checklist

1. Regenerate the exact staged tree with
   `PYTHONDONTWRITEBYTECODE=1 python scripts/stage_webhumanbench_v1_analysis_release.py --refresh`.
2. Push the contents of this package's `code/` directory to the anonymous
   repository without changing the staged tree.
3. From the anonymous repository root, run
   `python scripts/preflight_public_release.py --release-root . --output
   ../webhumanbench_v1_release_preflight.json`. Keep all generated reports
   outside the repository so the audited upload tree remains unchanged.
4. Run `python -m pytest -q` and the manifest-only audit in `README.md`.
5. Verify the public repository has no author identity, absolute local paths,
   credentials, raw restricted artifacts, or externally visible draft history.
6. Only after the repository is reachable, add its real URL and release date
   to `CITATION.cff` and the paper's availability statement.

No repository URL or release date is asserted here because neither should be
invented before the anonymous deposit exists.
