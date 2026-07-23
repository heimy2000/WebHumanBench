# Formal Benchmark Coverage Gate

WebHumanBench v1.0.0 is a locally frozen six-type release candidate. Its
executable formal-coverage audit returns `pilot_only`; it must not be described
as the formal multi-type WebHumanBench profile or a mature leaderboard.

## Required Coverage

Every canonical page type must have validated, group-disjoint scoring records:

| Group type | Per page type | Six-type total |
| --- | ---: | ---: |
| Historical-reference train | 120 | 720 |
| Historical-reference development | 40 | 240 |
| Historical-reference test | 40 | 240 |
| AI test | 100 | 600 |

The human-source target is 1,200 groups across `saas_landing`,
`docs_homepage`, `product_showcase`, `developer_tool`, `dashboard_shell`, and
`portfolio_showcase`.

Only records in a validated `webhumanbench_manifest.json` count. Candidate
repositories, discovery queues, pending reviews, source audits, partial
materializations, and partial captures never count. This coverage gate does not
prove authorship, preference, accessibility, regional representativeness, or
redistribution rights.

## V1 Status

V1 has 32 historical-reference sources with page-type counts 6/8/4/4/4/6 and
590 generated test groups. Its split has 16 reference train, 6 reference dev,
and 10 reference test groups. It remains far below the formal coverage profile.
The status is intentionally preserved in
`results/webhumanbench_v1_readiness.json`.

## Run the Audit

```bash
mkdir -p ../webhumanbench-recomputed
python scripts/audit_benchmark_readiness.py \
  --manifest benchmark/webhumanbench_v1/webhumanbench_manifest.json \
  --output ../webhumanbench-recomputed/webhumanbench_v1_readiness_recomputed.json
```

Only a future manifest that returns `formal_benchmark_ready`, passes artifact
audit, completes its rights review, and supports its claimed scope with
independent corpus coverage may be described as a formal WebHumanBench benchmark.
