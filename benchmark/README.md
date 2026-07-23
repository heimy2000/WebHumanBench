# WebHumanBench Benchmark Layout

## V1.0.0 Release Candidate

`webhumanbench_v1/` is the current locally frozen candidate package. The
anonymous `1.0.0-rc5` analysis release retains the files below except
`captures/`; it keeps their ledger paths and SHA-256 values but does not
redistribute raw materials subject to third-party or provider terms.

```text
benchmark/webhumanbench_v1/
|-- source_manifest.json
|-- corpus_assembly_receipt.json
|-- capture_rebind_receipt.json
|-- source_receipts.json
|-- closure_manifests/
|-- capture_ledger.json
|-- split_assignments.json
|-- ai_records.json
|-- ai_source_run_ledger.json
|-- ai_exclusion_ledger.json
|-- webhumanbench_manifest.json
`-- captures/
    |-- html/
    |-- screenshots/
    |-- snapshot_overlays/
    `-- ai/
```

The package contains 32 fixed-commit historical-reference sources across all
six canonical page types, 64 mobile captures, a 16/6/10 source-group split,
590 frozen generated challenge groups, and 622 scoring records. The two AI
ledgers close the 600-response source run as 590 retained plus 10 pre-score
offline-closure exclusions. `source_manifest.json` and
`corpus_assembly_receipt.json` bind the final source selection to the candidate
screen, product-recovery screen, and prior portfolio package. The
capture ledger and `capture_rebind_receipt.json` bind retained source closure
receipts and frozen vendor overlays to the final source-manifest digest. The
candidate excludes an upstream capture that failed the conservative token
preflight rather than redacting its contents.

Sixteen reference sources use 163 remote assets frozen at 2026 capture time,
so the fixed historical-commit policy applies to repository provenance rather
than every rendered asset. All retained generation prompts are extractor-aware and request
literal CSS plus a plausible modern human-authored style. See
`results/webhumanbench_v1_protocol_audit.json` for the executable audit.

Files and directories named `candidate_*`, `product_recovery_*`, or `*_rN` in
the working directory are construction evidence. They are not part of a staged
v1 candidate unless explicitly included in the source selection receipt.

## Verify v1

```bash
python scripts/audit_public_release.py \
  --source-manifest benchmark/webhumanbench_v1/source_manifest.json \
  --capture-ledger benchmark/webhumanbench_v1/capture_ledger.json \
  --benchmark-manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --output results/webhumanbench_v1_public_release_audit_recomputed.json

python scripts/audit_benchmark_readiness.py \
  --manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --output results/webhumanbench_v1_readiness_recomputed.json
```

The optional `--artifact-root` argument is only for an authorized full
provenance bundle containing the restricted raw captures. The first command
must pass before redistribution. The second command currently
returns `pilot_only`; that status is an explicit candidate boundary, not an error
to suppress.

## Build a Successor Version

Do not overwrite v1. A successor should use a new version directory and follow
this order:

1. Freeze reviewed source rows and an immutable source-selection receipt.
2. Materialize every selected fixed-commit entrypoint closure.
3. Capture both mobile viewports from local snapshots and require a complete
   render-integrity screen.
4. Generate a deterministic source-level split.
5. Freeze AI records whose feature-extractor version matches the human capture.
6. Copy only manifest-reachable generated artifacts and build the manifest.
7. Build source-run and exclusion ledgers, then run protocol, source-resplit,
   leave-one-reference-out, aggregation, challenge-composition, and measurement
   sensitivity audits.
8. Run artifact audit, baselines, readiness audit, staging, and preflight.

The historical-evidence source schema records pinned Git history plus
source-project identity and a fixed-commit provenance policy. It is a revision
traceability record, not a visual human-authorship label. A non-historical source schema
requires at least two independent provenance reviews instead.

## Formal Coverage Profile

The executable formal profile requires, for every page type, 120 reference
train, 40 reference dev, 40 reference test, and 100 generated test groups. That
is 1,200 historical-reference sources and 600 generated groups across all six types. Candidate
queues, failed captures, and unreviewed sources do not count.

See `docs/WEBHUMANBENCH_V1_RELEASE.md` for the v1 release boundary and
`docs/FORMAL_BENCHMARK_GATE.md` for the formal profile.
