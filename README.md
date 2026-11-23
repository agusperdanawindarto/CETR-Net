# CETR-Net — Counterfactual Evidence-Constrained Topology Refinement

**Executed research evidence | FIVES V2 | audit 23 November 2026 | manuscript format IEEE Access**

> **Negative primary result.** On the locked FIVES V2 official held-out test, the predeclared \`dual_full_asymmetric\` CETR-Net configuration did **not** outperform the independent backbone: mean Dice 0.849389 versus 0.851540, mean difference -0.002152 over seeds 42, 123 and 2025. The code and numeric evidence are archived without post-test model selection.

## Coverage

| Experiment | Audited status |
|---|---|
| Independently trained backbones (42/123/2025) | 3/3 completed; 45 epochs per seed |
| CDES/routing full-data refiners | 6/6 training groups completed, 21/21 seed-variant combinations, 8 epochs each |
| Official held-out test (FIVES V2) | 3/3 seeds, 200 annotation records, 197 screened image groups |
| Mean backbone Dice / IoU / clDice | 0.851540 / 0.749034 / 0.871657 |
| Mean primary proposed Dice / IoU / clDice | 0.849389 / 0.745639 / 0.870490 |
| External cross-dataset transfer | NOT RUN; no generalization claim |
| Matched-capacity MaskVSC / U-Net / Attention U-Net | NOT RUN; no SOTA claim |
| Full candidate-level topology-error benchmark or complexity analysis | NOT COMPLETE; cannot claim mechanistic superiority |

## Protocol (frozen before official test)

- FIVES source: 800 annotated images, official development/test 600/200.
- **Actual audited V2 development:** 508 train + 90 validation per seed, after exclusions from the historical 600-record partition.
- Official test: 200 annotation records aggregated over 197 visual groups (not 197 independent patients).
- Best checkpoints selected from validation only; threshold 0.5.
- No ground-truth labels fed into the candidate generator during validation or test.
- Primary comparison was \`dual_full_asymmetric\` versus the frozen independent backbone.
- Per-seed primary uncertainty: 2,000 paired bootstrap resamples over the 197 visual groups.
- No test-based variant selection, threshold fitting, or post-hoc primary switching.

## Contents

- \`scripts/\`: source snapshots of training, topology refinement, split screening, Kaggle evaluation, and aggregation, as archived from the execution workspace.
- \`notebooks/\`: notebook source snapshots with outputs stripped; input Kaggle datasets must be connected by an authorized user.
- \`results/official_fives_v2_cross_seed_table.csv\`: official mean Dice, IoU, clDice and seed variability for eight methods.
- \`results/per_seed/\`: per-seed verification/checksums and groupwise/per-record numerical metrics (not images/masks).
- \`results/refinement/\`: all six finished CDES and routing groups and checkpoint hashes.
- \`protocol/frozen_fives_v2_test_plan.json\`: pre-test selected primary, secondary variants, score and bootstrap rules.
- \`docs/\`: research interpretation and reproduction notes.

### Primary held-out comparison

| Seed | Independent backbone | Proposed dual full asymmetric | Delta Dice |
|---|---:|---:|---:|
| 42 | 0.853179 | 0.850975 | -0.002205 |
| 123 | 0.850769 | 0.848570 | -0.002199 |
| 2025 | 0.850673 | 0.848621 | -0.002052 |
| **Mean** | **0.851540** | **0.849389** | **-0.002152** |

## Integrity and ethical limitations

The independent model backbone is trained separately for each seed. Routing/head training uses the frozen, seed-matched backbone. The completed experiments test a **head-only refinement** mechanism; they do not constitute an end-to-end jointly trained novel architecture. The presence of 197 visually distinct groups does not establish patient-disjoint acquisitions. No external dataset, cross-device generalization, clinical diagnostic trial, independent SOTA replication, or comprehensive capacity/FLOPs benchmark was completed.

**Publishing note:** FIVES raw RGB photographs, binary reference masks, Kaggle private datasets, user credentials, checkpoint binaries and any sensitive original records are intentionally excluded. Download the licensed FIVES source separately and verify hashes before reuse. The official test must not be revisited for parameter selection after seeing these outcomes.

**IEEE manuscript:** The manuscript may report the *negative evidence* but should not claim superiority. Source citations and prior-method bibliographic entries require editorial audit before submission.

## Top-tier IEEE revision and source-traceable FIVES results

Revised manuscript and Mendeley-ready BibTeX: manuscript/. Research integrity audit: docs/TOP_TIER_V4_AUDIT.md. Three real fixed-ID qualitative FIVES test panels are archived in figures/qualitative/, attributable to Jin et al. (2022), CC BY 4.0. Full underlying raw dataset archives and private masks remain excluded. The updated paper does NOT claim superiority. Live Mendeley fields require the author's signed-in Word Mendeley Cite session; static manuscript citations are NOT dynamically linked.

## Current manuscript (Top Tier V5, 23 November 2026)

The [IEEE Access Word V5](manuscript/CETR-Net_IEEE_TopTier_Revised_V5.docx) and [PDF proof V5](manuscript/CETR-Net_IEEE_TopTier_Revised_V5.pdf) resolve stale RQ3 cross-dataset/efficiency statements and reconcile native Eq. (11) with the 197-group aggregation rule. The [V5 research audit](docs/TOP_TIER_V5_AUDIT.md) documents verified results and outstanding Mendeley Cite linkage. Quantitative evidence and original FIVES qualitative figures are unchanged from V4.
