# Third-Party Artifact Notices

## Scope

The v1.0.0-rc5 anonymous analysis release includes ledgers, hashes, derived
features, frozen results, and code. It deliberately excludes frozen rendered
captures, entrypoint closures, local snapshot overlays, generated HTML, raw
provider responses, and screenshots. This notice records the rights boundary;
it does not grant rights in third-party material.

- Repository implementation code is Apache-2.0.
- Benchmark metadata and derived feature arrays are CC-BY-4.0.
- Captured source pages, entrypoint closures, snapshot overlays, vendored fonts,
  icons, scripts, images, generated HTML, raw model responses, and screenshots
  remain excluded and retain their upstream, author, asset-owner, or provider
  terms.
- Reusers are responsible for checking those terms before requesting or
  redistributing a capture, source-derived overlay, or model artifact.

## Source Projects

The 32 fixed-commit source repositories are canonically recorded in
`benchmark/webhumanbench_v1/source_manifest.json`. At the selected commits the
manifest records 24 MIT, 5 Apache-2.0, 2 BSD-3-Clause, and 1 CC0-1.0 source
licenses. The source manifest, not this summary, is authoritative for each
repository URL, commit, entrypoint, license evidence, and temporal provenance
evidence.

The v1 corpus contains six page-type strata. A historical open-source evidence
record is a release-provenance proxy; it is not a claim that an individual
creator is human or that a page was produced without assistance.

## Frozen External Assets

The v1 capture ledger records the original HTTPS URL, content type, local
hash-addressed path, and SHA-256 for every vendored visual asset. It also
records narrowly removed nonvisual tracking and legacy browser-compatibility
scripts. The removal policy preserves Chromium rendering behavior for the
declared capture protocol; it does not relicense or otherwise alter the rights
in the upstream resource.

Observed providers can include font CDNs, icon libraries, framework assets, and
project-specific image hosts. Provider names identify observed dependencies,
not a common license for all files served by a provider.

## Model Artifacts

The released AI ledger includes feature records, configurations, artifact paths,
and SHA-256 values. Raw provider responses, generated HTML, offline-rendered
HTML, and screenshots are excluded. They remain subject to applicable provider
and model terms, and any later access or use does not imply provider endorsement.

For exact artifact-level attribution, inspect
`benchmark/webhumanbench_v1/capture_ledger.json` and each AI record's
`provenance.artifacts` object in `benchmark/webhumanbench_v1/ai_records.json`.
